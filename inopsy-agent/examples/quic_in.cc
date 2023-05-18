/*
 * ngtcp2
 *
 * Copyright (c) 2021 ngtcp2 contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <linux/if_tun.h>
#include <netinet/tcp.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <variant>
#include <optional>
#include <queue>
#include <random>

#include <ngtcp2/ngtcp2.h>
#include <ngtcp2/ngtcp2_crypto.h>
#include <ngtcp2/ngtcp2_crypto_openssl.h>
#include <ngtcp2/fec_callbacks.h>
#include <ngtcp2/fec_r_scheme.h>
#include <ngtcp2/fec_leopard.h>
#include <ngtcp2/fec_reed_sol.h>
#include <ngtcp2/fec_1pr.h>
#include <ngtcp2/fec_2pr.h>
#include <ngtcp2/fec_dummy.h>

#include <openssl/ssl.h>
#include <openssl/rand.h>
#include <openssl/err.h>

#include <ev.h>

#include "CLI11.hpp"
#include "tunnel_utils.h"
#include "util.h"
#include "time_logger.h"

#define ALPN "\xaperf"

const int DEFAULT_MTU = 1500;
const int PKT_MAX_SIZE = 1280; // value from simpleclient

static uint64_t timestamp(void) {
  struct timespec tp;

  if (clock_gettime(CLOCK_MONOTONIC, &tp) != 0) {
    fprintf(stderr, "clock_gettime: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  return (uint64_t)tp.tv_sec * NGTCP2_SECONDS + (uint64_t)tp.tv_nsec;
}

static int create_sock(struct sockaddr *addr, socklen_t *paddrlen,
                       const char *host, const char *port) {
  struct addrinfo hints = {0};
  struct addrinfo *res, *rp;
  int rv;
  int fd = -1;

  hints.ai_flags = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;

  rv = getaddrinfo(host, port, &hints, &res);
  if (rv != 0) {
    fprintf(stderr, "getaddrinfo: %s %s %s !\n", gai_strerror(rv), host, port);
    return -1;
  }

  for (rp = res; rp; rp = rp->ai_next) {
    fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) {
      continue;
    }

    break;
  }

  if (fd == -1) {
    goto end;
  }

  *paddrlen = rp->ai_addrlen;
  memcpy(addr, rp->ai_addr, rp->ai_addrlen);

end:
  freeaddrinfo(res);

  return fd;
}

static int connect_sock(struct sockaddr *local_addr, socklen_t *plocal_addrlen,
                        int fd, const struct sockaddr *remote_addr,
                        size_t remote_addrlen) {
  socklen_t len;

  while (connect(fd, remote_addr, (socklen_t)remote_addrlen) != 0) {
    fprintf(stderr, "connect: %s\n", strerror(errno));
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  len = *plocal_addrlen;

  if (getsockname(fd, local_addr, &len) == -1) {
    fprintf(stderr, "getsockname: %s\n", strerror(errno));
    return -1;
  }

  *plocal_addrlen = len;

  return 0;
}

struct QBuf {
  std::vector<uint8_t> data;
  // stream offset of next byte after |data|
  size_t offset;
};

struct client {
  int fd;
  int tun_fd;
  struct sockaddr_storage local_addr;
  socklen_t local_addrlen;
  SSL_CTX *ssl_ctx;
  SSL *ssl;
  ngtcp2_conn *conn;
  int64_t stream_id; // stream for packets

  struct {
    int64_t stream_id;
    const uint8_t *data;
    size_t datalen;
    size_t nwrite;
  } stream;

  uint64_t last_error;

  ev_io rev;
  ev_io tun_watcher;
  ev_timer timer;
  // timer for sending fec_buf if we stopped receiving packets
  ev_timer send_timer;

  char remote_host[20];
  char remote_port[20];

  void* fec_cfg;
  void* fec_state;
  fec_callbacks* fec_cb;
  // buffer of data for original and recovery packets
  // memory layout: [original][recovery]
  std::vector<uint8_t> fec_buf;
  // current offset in buffer of info packets
  size_t fec_buf_offset = 0;
  
  size_t sendq_buf_offset = 0;
  std::queue<std::vector<uint8_t>> sendq;
  // we hold data for possible retransmits until acknowledged
  std::queue<QBuf> storage;

  uint64_t bytes_rcvd = 0;
  uint64_t bytes_sent = 0;

  std::string dump_file = {};
  bool disable_logging = false;
  // fraction of lost packets for loss emulation 
  double loss = 0.0;
  // when fin mode is active every packet is extended into chunk with zeroes
  // pros: packet is being sent almost immediately
  // cons: large overhead: |size of chunk| - |size of packet| bytes 
  bool fin_mode = false;
  // whether to enable fin mode when encountering TCP FIN packet
  // allows sending last packets of TCP session without waiting to accumulate whole chunks
  bool enable_fin_mode = false;
  // whether use inopsy-type logs or not
  bool inopsy_log_is_set = true;
  // congestion control algorithm
  ngtcp2_cc_algo cc_algo;
  // BBRv2 parameters:
  double bbr2_loss_tresh;
  double bbr2_beta;
  double bbr2_probe_rtt_cwnd_gain;
  ngtcp2_duration bbr2_probe_rtt_duration;

  int init_fec(void* cfg, fec_callbacks* cb)
  {
    fec_cfg = cfg;
    fec_cb = cb;

    if (cb->init) {
      auto ret = cb->init(cfg, &fec_state);
      if (ret) { return ret; }
    }
    
    // reserve space for both original and recovery data
    fec_buf.resize((cb->num_original(cfg) + cb->num_recovery(cfg)) * cb->payload_size(cfg));
    
    return 0;
  }
};



static int set_encryption_secrets(SSL *ssl, OSSL_ENCRYPTION_LEVEL ossl_level,
                                  const uint8_t *rx_secret,
                                  const uint8_t *tx_secret, size_t secretlen) {
  struct client *c = (client*)SSL_get_app_data(ssl);
  ngtcp2_crypto_level level =
      ngtcp2_crypto_openssl_from_ossl_encryption_level(ossl_level);

  if (rx_secret &&
      ngtcp2_crypto_derive_and_install_rx_key(c->conn, NULL, NULL, NULL, level,
                                              rx_secret, secretlen) != 0) {
    fprintf(stderr, "ngtcp2_crypto_derive_and_install_rx_key failed\n");
    return 0;
  }

  if (ngtcp2_crypto_derive_and_install_tx_key(c->conn, NULL, NULL, NULL, level,
                                              tx_secret, secretlen) != 0) {
    fprintf(stderr, "ngtcp2_crypto_derive_and_install_tx_key failed\n");
    return 0;
  }

  return 1;
}

static int add_handshake_data(SSL *ssl, OSSL_ENCRYPTION_LEVEL ossl_level,
                              const uint8_t *data, size_t len) {
  struct client *c = (client*)SSL_get_app_data(ssl);
  ngtcp2_crypto_level level =
      ngtcp2_crypto_openssl_from_ossl_encryption_level(ossl_level);
  int rv;

  rv = ngtcp2_conn_submit_crypto_data(c->conn, level, data, len);
  if (rv != 0) {
    fprintf(stderr, "ngtcp2_conn_submit_crypto_data: %s\n",
            ngtcp2_strerror(rv));
    return 0;
  }

  return 1;
}

static int flush_flight(SSL *ssl) {
  (void)ssl;
  return 1;
}

static int send_alert(SSL *ssl, OSSL_ENCRYPTION_LEVEL ossl_level,
                      uint8_t alert) {
  struct client *c = (client*)SSL_get_app_data(ssl);
  (void)ossl_level;

  c->last_error = NGTCP2_CRYPTO_ERROR | alert;

  return 1;
}

static SSL_QUIC_METHOD quic_method = {
    set_encryption_secrets,
    add_handshake_data,
    flush_flight,
    send_alert,
};

static int numeric_host_family(const char *hostname, int family) {
  uint8_t dst[sizeof(struct in6_addr)];
  return inet_pton(family, hostname, dst) == 1;
}

static int numeric_host(const char *hostname) {
  return numeric_host_family(hostname, AF_INET) ||
         numeric_host_family(hostname, AF_INET6);
}

static int client_ssl_init(struct client *c) {
  c->ssl_ctx = SSL_CTX_new(TLS_client_method());
  if (!c->ssl_ctx) {
    fprintf(stderr, "SSL_CTX_new: %s\n",
            ERR_error_string(ERR_get_error(), NULL));
    return -1;
  }

  SSL_CTX_set_min_proto_version(c->ssl_ctx, TLS1_3_VERSION);
  SSL_CTX_set_max_proto_version(c->ssl_ctx, TLS1_3_VERSION);
  SSL_CTX_set_quic_method(c->ssl_ctx, &quic_method);

  c->ssl = SSL_new(c->ssl_ctx);
  if (!c->ssl) {
    fprintf(stderr, "SSL_new: %s\n", ERR_error_string(ERR_get_error(), NULL));
    return -1;
  }

  SSL_set_app_data(c->ssl, c);
  SSL_set_connect_state(c->ssl);
  SSL_set_alpn_protos(c->ssl, (const unsigned char *)ALPN, sizeof(ALPN) - 1);
  if (!numeric_host(c->remote_host)) {
    SSL_set_tlsext_host_name(c->ssl, c->remote_host);
  }

  return 0;
}

static void rand_cb(uint8_t *dest, size_t destlen,
                    const ngtcp2_rand_ctx *rand_ctx) {
  size_t i;
  (void)rand_ctx;

  for (i = 0; i < destlen; ++i) {
    *dest = (uint8_t)random();
  }
}

/**
 * Erase acknowledged data from storage
 */
static int acked_stream_data_offset(
  ngtcp2_conn *conn, int64_t stream_id, uint64_t offset, uint64_t datalen, 
  void *user_data, void *stream_user_data
) {
  client* c = static_cast<client*>(user_data);

  while (!c->storage.empty()) {
    auto& buf = c->storage.front();
    if (buf.offset < offset + datalen) {
      c->storage.pop();
    } else {
      break;
    }
  }
  return 0;
}

static int get_new_connection_id_cb(ngtcp2_conn *conn, ngtcp2_cid *cid,
                                    uint8_t *token, size_t cidlen,
                                    void *user_data) {
  (void)conn;
  (void)user_data;

  if (RAND_bytes(cid->data, (int)cidlen) != 1) {
    return NGTCP2_ERR_CALLBACK_FAILURE;
  }

  cid->datalen = cidlen;

  if (RAND_bytes(token, NGTCP2_STATELESS_RESET_TOKENLEN) != 1) {
    return NGTCP2_ERR_CALLBACK_FAILURE;
  }

  return 0;
}

static int extend_max_local_streams_bidi(ngtcp2_conn *conn,
                                         uint64_t max_streams,
                                         void *user_data) {
#ifdef MESSAGE
  struct client *c = (client*)user_data;
  int rv;
  int64_t stream_id;
  (void)max_streams;

  if (c->stream.stream_id != -1) {
    return 0;
  }

  rv = ngtcp2_conn_open_bidi_stream(conn, &stream_id, NULL);
  if (rv != 0) {
    return 0;
  }

  c->stream.stream_id = stream_id;
  c->stream.data = (const uint8_t *)MESSAGE;
  c->stream.datalen = sizeof(MESSAGE) - 1;

  return 0;
#else  /* !MESSAGE */
  (void)conn;
  (void)max_streams;
  (void)user_data;

  return 0;
#endif /* !MESSAGE */
}

static void log_printf(void *user_data, const char *fmt, ...) {
  va_list ap;
  (void)user_data;

  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  fprintf(stderr, "\n");
}

static int client_quic_init(struct client *c,
                            const struct sockaddr *remote_addr,
                            socklen_t remote_addrlen,
                            const struct sockaddr *local_addr,
                            socklen_t local_addrlen) {
  ngtcp2_path path = {
      {
          (struct sockaddr *)local_addr,
          local_addrlen,
      },
      {
          (struct sockaddr *)remote_addr,
          remote_addrlen,
      },
      NULL,
  };
  ngtcp2_callbacks callbacks = {
      ngtcp2_crypto_client_initial_cb,
      NULL, /* recv_client_initial */
      ngtcp2_crypto_recv_crypto_data_cb,
      NULL, /* handshake_completed */
      NULL, /* recv_version_negotiation */
      ngtcp2_crypto_encrypt_cb,
      ngtcp2_crypto_decrypt_cb,
      ngtcp2_crypto_hp_mask_cb,
      NULL, /* recv_stream_data */
      acked_stream_data_offset,
      NULL, /* stream_open */
      NULL, /* stream_close */
      NULL, /* recv_stateless_reset */
      ngtcp2_crypto_recv_retry_cb,
      extend_max_local_streams_bidi,
      NULL, /* extend_max_local_streams_uni */
      rand_cb,
      get_new_connection_id_cb,
      NULL, /* remove_connection_id */
      ngtcp2_crypto_update_key_cb,
      NULL, /* path_validation */
      NULL, /* select_preferred_address */
      NULL, /* stream_reset */
      NULL, /* extend_max_remote_streams_bidi */
      NULL, /* extend_max_remote_streams_uni */
      NULL, /* extend_max_stream_data */
      NULL, /* dcid_status */
      NULL, /* handshake_confirmed */
      NULL, /* recv_new_token */
      ngtcp2_crypto_delete_crypto_aead_ctx_cb,
      ngtcp2_crypto_delete_crypto_cipher_ctx_cb,
      NULL, /* recv_datagram */
      NULL, /* ack_datagram */
      NULL, /* lost_datagram */
      ngtcp2_crypto_get_path_challenge_data_cb,
      NULL, /* stream_stop_sending */
  };
  ngtcp2_cid dcid, scid;
  ngtcp2_settings settings;
  ngtcp2_transport_params params;
  int rv;

  dcid.datalen = NGTCP2_MIN_INITIAL_DCIDLEN;
  if (RAND_bytes(dcid.data, (int)dcid.datalen) != 1) {
    fprintf(stderr, "RAND_bytes failed\n");
    return -1;
  }

  scid.datalen = 8;
  if (RAND_bytes(scid.data, (int)scid.datalen) != 1) {
    fprintf(stderr, "RAND_bytes failed\n");
    return -1;
  }

  ngtcp2_settings_default(&settings);

  settings.initial_ts = timestamp();
  if (!c->disable_logging) {
    settings.log_printf = log_printf;
  }
  settings.cc_algo = c->cc_algo;

  ngtcp2_transport_params_default(&params);

  params.initial_max_streams_uni = 3;
  params.initial_max_stream_data_bidi_local = 128 * 1024;
  params.initial_max_data = 1024 * 1024;

  params.bbr2_loss_tresh = c->bbr2_loss_tresh;
  params.bbr2_beta = c->bbr2_beta;
  params.bbr2_probe_rtt_cwnd_gain = c->bbr2_probe_rtt_cwnd_gain;
  params.bbr2_probe_rtt_duration = c->bbr2_probe_rtt_duration;

  rv =
      ngtcp2_conn_client_new(&c->conn, &dcid, &scid, &path, NGTCP2_PROTO_VER_V1,
                             &callbacks, &settings, &params, NULL, c);
  if (rv != 0) {
    fprintf(stderr, "ngtcp2_conn_client_new: %s\n", ngtcp2_strerror(rv));
    return -1;
  }

  ngtcp2_conn_set_tls_native_handle(c->conn, c->ssl);

  return 0;
}

static int client_read(struct client *c) {
  uint8_t buf[65536];
  struct sockaddr_storage addr;
  struct iovec iov = {buf, sizeof(buf)};
  struct msghdr msg = {0};
  ssize_t nread;
  ngtcp2_path path;
  ngtcp2_pkt_info pi = {0};
  int rv;

  msg.msg_name = &addr;
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  for (;;) {
    msg.msg_namelen = sizeof(addr);

    nread = recvmsg(c->fd, &msg, MSG_DONTWAIT);

    if (nread == -1) {
      if (errno != EAGAIN && errno != EWOULDBLOCK) {
        fprintf(stderr, "recvmsg: %s\n", strerror(errno));
      }

      break;
    }

    if (c->inopsy_log_is_set) {
      fprintf(stderr, "!ReceivedBytes:%ld\n", (long)nread);
    }

    path.local.addrlen = c->local_addrlen;
    path.local.addr = (struct sockaddr *)&c->local_addr;
    path.remote.addrlen = msg.msg_namelen;
    path.remote.addr = (ngtcp2_sockaddr*)msg.msg_name;

    rv = ngtcp2_conn_read_pkt(c->conn, &path, &pi, buf, (size_t)nread,
                              timestamp());
    if (rv != 0) {
      fprintf(stderr, "ngtcp2_conn_read_pkt: %s\n", ngtcp2_strerror(rv));
      switch (rv) {
      case NGTCP2_ERR_REQUIRED_TRANSPORT_PARAM:
      case NGTCP2_ERR_MALFORMED_TRANSPORT_PARAM:
      case NGTCP2_ERR_TRANSPORT_PARAM:
      case NGTCP2_ERR_PROTO:
        c->last_error = ngtcp2_err_infer_quic_transport_error_code(rv);
        break;
      default:
        if (!c->last_error) {
          c->last_error = ngtcp2_err_infer_quic_transport_error_code(rv);
        }
        break;
      }
      return -1;
    }
  }

  return 0;
}

std::mt19937 make_mt19937() {
  std::random_device rd;
  return std::mt19937(rd());
}

auto randgen = make_mt19937();

bool packet_lost(double prob) {
  auto p = std::uniform_real_distribution<>(0, 1)(randgen);
  return p < prob;
}

static int client_send_packet(struct client *c, const uint8_t *data,
                              size_t datalen) {
  if (c->loss && packet_lost(c->loss)) {
    std::cerr << "** Simulated outgoing packet loss **" << std::endl;
    return 0;
  }

  struct iovec iov = {(uint8_t *)data, datalen};
  struct msghdr msg = {0};
  ssize_t nwrite;

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  do {
    nwrite = sendmsg(c->fd, &msg, 0);
  } while (nwrite == -1 && errno == EINTR);

  if (nwrite == -1) {
    fprintf(stderr, "sendmsg: %s\n", strerror(errno));

    return -1;
  }

  if (c->inopsy_log_is_set) {
    fprintf(stderr, "!SentBytes:%ld\n", (long)nwrite);
  }

  return 0;
}

static size_t client_get_message(struct client *c, int64_t *pstream_id,
                                 int *pfin, ngtcp2_vec *datav,
                                 size_t datavcnt) {
  if (datavcnt == 0) {
    return 0;
  }

  *pstream_id = -1;
  *pfin = 0;
  datav->base = NULL;
  datav->len = 0;

  return 0;
}

static int client_write_streams(struct client *c) {
  ngtcp2_tstamp ts = timestamp();
  ngtcp2_pkt_info pi;
  ngtcp2_ssize nwrite;
  uint8_t buf[1280];
  ngtcp2_path_storage ps;
  ngtcp2_vec datav;
  size_t datavcnt;
  int64_t stream_id;
  ngtcp2_ssize wdatalen;
  uint32_t flags;
  int fin;

  ngtcp2_path_storage_zero(&ps);

  for (;;) {
    datavcnt = client_get_message(c, &stream_id, &fin, &datav, 1);

    flags = NGTCP2_WRITE_STREAM_FLAG_MORE;
    if (fin) {
      flags |= NGTCP2_WRITE_STREAM_FLAG_FIN;
    }

    nwrite = ngtcp2_conn_writev_stream(c->conn, &ps.path, &pi, buf, sizeof(buf),
                                       &wdatalen, flags, stream_id, &datav,
                                       datavcnt, ts);
    if (nwrite < 0) {
      switch (nwrite) {
      case NGTCP2_ERR_WRITE_MORE:
        c->stream.nwrite += (size_t)wdatalen;
        continue;
      default:
        fprintf(stderr, "ngtcp2_conn_writev_stream: %s\n",
                ngtcp2_strerror((int)nwrite));
        c->last_error = ngtcp2_err_infer_quic_transport_error_code((int)nwrite);
        return -1;
      }
    }

    if (nwrite == 0) {
      return 0;
    }

    if (wdatalen > 0) {
      c->stream.nwrite += (size_t)wdatalen;
    }

    if (client_send_packet(c, buf, (size_t)nwrite) != 0) {
      break;
    }
  }

  return 0;
}

static int client_write(struct client *c) {
  ngtcp2_tstamp expiry, now;
  ev_tstamp t;

  if (client_write_streams(c) != 0) {
    return -1;
  }

  expiry = ngtcp2_conn_get_expiry(c->conn);
  now = timestamp();

  t = expiry < now ? 1e-9 : (ev_tstamp)(expiry - now) / NGTCP2_SECONDS;

  c->timer.repeat = t;
  ev_timer_again(EV_DEFAULT, &c->timer);

  return 0;
}

static int client_handle_expiry(struct client *c) {
  int rv = ngtcp2_conn_handle_expiry(c->conn, timestamp());
  if (rv != 0) {
    fprintf(stderr, "ngtcp2_conn_handle_expiry: %s\n", ngtcp2_strerror(rv));
    return -1;
  }

  return 0;
}

static void client_close(struct client *c) {
  ngtcp2_ssize nwrite;
  ngtcp2_pkt_info pi;
  ngtcp2_path_storage ps;
  uint8_t buf[1280];

  if (ngtcp2_conn_is_in_closing_period(c->conn) || !c->last_error) {
    goto fin;
  }

  ngtcp2_path_storage_zero(&ps);

  nwrite = ngtcp2_conn_write_connection_close(c->conn, &ps.path, &pi, buf,
                                              sizeof(buf), c->last_error, NULL,
                                              0, timestamp());
  if (nwrite < 0) {
    fprintf(stderr, "ngtcp2_conn_write_connection_close: %s\n",
            ngtcp2_strerror((int)nwrite));
    goto fin;
  }

  client_send_packet(c, buf, (size_t)nwrite);

fin:
  ev_break(EV_DEFAULT, EVBREAK_ALL);
}

static void read_cb(struct ev_loop *loop, ev_io *w, int revents) {
  struct client *c = (client*)w->data;
  (void)loop;
  (void)revents;

  if (client_read(c) != 0) {
    client_close(c);
    return;
  }

  if (client_write(c) != 0) {
    client_close(c);
  }
}

static void timer_cb(struct ev_loop *loop, ev_timer *w, int revents) {
  struct client *c = (client*)w->data;
  (void)loop;
  (void)revents;

  if (client_handle_expiry(c) != 0) {
    client_close(c);
    return;
  }

  if (client_write(c) != 0) {
    client_close(c);
  }
}

void send_from_queue(client *c) {
  ngtcp2_tstamp ts = timestamp();
  ngtcp2_pkt_info pi;
  ngtcp2_ssize nwrite;
  ngtcp2_path_storage ps;
  ngtcp2_vec datav;
  size_t datavcnt = 1;
  int64_t stream_id;
  ngtcp2_ssize wdatalen;
  uint32_t flags;
  ngtcp2_path_storage_zero(&ps);

  if (!c->fec_cb) { return; }

  auto payload_size = c->fec_cb->payload_size(c->fec_cfg);
  while (!c->sendq.empty()) {
    uint8_t pkt_buf[PKT_MAX_SIZE];

    auto& sendbuf = c->sendq.front();
    auto& sendbuf_offset = c->sendq_buf_offset;
    while (sendbuf_offset < sendbuf.size()) {
      // In high loss/low speed scenarios frame can carry less than whole payload.
      // Try to send only remaining part of payload so that following frames
      // will be aligned correctly. 
      auto remaining_size = payload_size;
      auto payload_offset = sendbuf_offset % payload_size;
      if (payload_offset != 0) {
        remaining_size -= payload_offset;
      }

      datav.base = &sendbuf[sendbuf_offset];
      datav.len = ((sendbuf_offset + payload_size) > sendbuf.size()) 
        ? (sendbuf.size() - sendbuf_offset) : payload_size;
      datav.len = std::min(static_cast<uint64_t>(remaining_size), datav.len);

      // FIXME: stream_id is default-initialized
      nwrite = ngtcp2_conn_writev_stream(c->conn, &ps.path, &pi, pkt_buf, sizeof(pkt_buf),
        &wdatalen, flags, stream_id, &datav, datavcnt, ts);

      if (nwrite < 0) {
        fprintf(stderr, "ngtcp2_conn_writev_stream: %s\n",
                ngtcp2_strerror((int)nwrite));
        return;
      }

      if (nwrite == 0) { // need acks or extended stream size
        return;
      }

      if (wdatalen != -1) { 
        c->bytes_sent += wdatalen;
        sendbuf_offset += wdatalen;
      }

      if (client_send_packet(c, pkt_buf, (size_t)nwrite) != 0) {
        fprintf(stderr, "client_send_packet: %s\n",
                  ngtcp2_strerror((int)nwrite));
        return;
      }
    }
    c->storage.emplace(QBuf{});
    std::swap(c->storage.back().data, c->sendq.front());    
    c->storage.back().offset = c->bytes_sent;
    c->sendq.pop();
    c->sendq_buf_offset = 0;
  }
}

// hack to send first packet immediately: we need handshake to happen
// and never send non-filled FEC buffers afterwards
bool once = false;

static void send_timer_cb(struct ev_loop *loop, ev_timer *w, int revents) {
  (void)loop;
  (void)w;
  (void)revents;

  struct client *c = (client*)w->data;
  // fprintf(stderr, "send timeout, sent %lu/%lu\n", c->bytes_sent, c->bytes_rcvd);

  if (c->fec_buf_offset != 0 && !once) {
    once = true;
    c->sendq.push({});
    auto fec_buf_size = c->fec_buf.size();
    c->fec_buf.resize(c->fec_buf_offset); // buffer not full
    std::swap(c->fec_buf, c->sendq.back());
    c->fec_buf.resize(fec_buf_size);
    c->fec_buf_offset = 0;
  }
  send_from_queue(c);
}

void read_tun_cb(struct ev_loop *loop, ev_io *w, int revents) {
  int64_t stream_id;

  struct client *c = (client*)w->data;

  ev_timer_again(EV_DEFAULT, &c->send_timer);

  uint8_t buf[DEFAULT_MTU];
  auto rcvd = tun_read(c->tun_fd, (char*)buf, sizeof(buf));
  if ((buf[0] >> 4) != 4) { // ignore ipv6
    return;
  }
  // FIN is last bit in 13th byte (from 0) of TCP header
  // we expect IP header length of 20 bytes (no options)
  if (c->enable_fin_mode && !c->fin_mode && buf[33] % 2) {
    c->fin_mode = true;
  }
  c->bytes_rcvd += rcvd;

  if (c->stream_id == -1) {
    auto ret = ngtcp2_conn_open_bidi_stream(c->conn, &c->stream_id, nullptr);
    if (ret != 0) {
      std::cerr << "ngtcp2_conn_open_bidi_stream: " << ngtcp2_strerror(ret) << std::endl;
      return;
    }
  }

  if (!c->fec_cb) {
    // send immediately
    ngtcp2_tstamp ts = timestamp();
    ngtcp2_pkt_info pi;
    ngtcp2_ssize nwrite;
    ngtcp2_path_storage ps;
    ngtcp2_vec datav;
    size_t datavcnt = 1;
    ngtcp2_ssize wdatalen;
    uint32_t flags;
    ngtcp2_path_storage_zero(&ps);

    // std::cout << "resend message immediately" << std::endl;
    datav.base = buf;
    datav.len = rcvd;

    uint8_t pkt_buf[PKT_MAX_SIZE];

    nwrite = ngtcp2_conn_writev_stream(c->conn, &ps.path, &pi, pkt_buf, sizeof(pkt_buf),
      &wdatalen, flags, stream_id, &datav, datavcnt, ts);
    if (nwrite < 0) {
      fprintf(stderr, "ngtcp2_conn_writev_stream: %s\n",
              ngtcp2_strerror((int)nwrite));
      return;
    }

    if (nwrite == 0) {
      return;
    }

    // std::cout << "send packet of size " << nwrite << std::endl;
    if (client_send_packet(c, pkt_buf, (size_t)nwrite) != 0) {
      fprintf(stderr, "client_send_packet: %s\n",
                ngtcp2_strerror((int)nwrite));
      return;
    }
    c->bytes_sent += rcvd;
  } else {
    // collect data into buffer
    uint32_t fec_original_size = \
      c->fec_cb->num_original(c->fec_cfg) * c->fec_cb->payload_size(c->fec_cfg);
    size_t rcvd_offset = (c->fec_buf_offset + rcvd <= fec_original_size) 
      ? rcvd : fec_original_size - c->fec_buf_offset;
    memcpy(&(c->fec_buf[c->fec_buf_offset]), buf, rcvd_offset);
    c->fec_buf_offset += rcvd_offset;
    // std::cout << "accumulate " << c->fec_buf_offset << " bytes" << std::endl;

    if (c->fin_mode || c->fec_buf_offset == fec_original_size) {
      if (c->fin_mode) {
        memset(&c->fec_buf[c->fec_buf_offset], 0, fec_original_size - c->fec_buf_offset);
        c->fec_buf_offset = fec_original_size;
      }
      auto ret = c->fec_cb->encode(
        c->fec_cfg, c->fec_state, &c->fec_buf[0], 
        &c->fec_buf[fec_original_size]
      );
      if (ret) {
        throw std::runtime_error("failed to encode data");
      }

      // write to file
      if (!c->dump_file.empty()) {
        std::ofstream file;
        file.open(c->dump_file, std::ios_base::app);
        if (file.fail()) {
          throw std::runtime_error("cannot write to file");
        }
        for (int i = 0; i < c->fec_buf.size(); ++i) {
          if (i % 8 == 0) { // make formatting more readable
            file << "\n";
          }
          file << int(c->fec_buf[i]) << " ";
        }
        file << "\n"; // extra newline to separate FEC groups
      }
      // add to sendq
      c->sendq.push({});
      auto fec_buf_size = c->fec_buf.size();
      std::swap(c->fec_buf, c->sendq.back());
      c->fec_buf.resize(fec_buf_size);
      c->fec_buf_offset = 0;
      // std::cout << "push queue, new size " << c->sendq.size() << std::endl;
    }
    if (rcvd_offset < rcvd) {
      memcpy(&(c->fec_buf[c->fec_buf_offset]), buf + rcvd_offset, rcvd - rcvd_offset);
      c->fec_buf_offset += rcvd - rcvd_offset;
    }

    send_from_queue(c);
  }
}

static int client_init(struct client *c) {
  struct sockaddr_storage remote_addr, local_addr;
  socklen_t remote_addrlen, local_addrlen = sizeof(local_addr);

  c->fd = create_sock((struct sockaddr *)&remote_addr, &remote_addrlen,
                      c->remote_host, c->remote_port);
  if (c->fd == -1) {
    return -1;
  }

  if (connect_sock((struct sockaddr *)&local_addr, &local_addrlen, c->fd,
                   (struct sockaddr *)&remote_addr, remote_addrlen) != 0) {
    return -1;
  }

  memcpy(&c->local_addr, &local_addr, sizeof(c->local_addr));
  c->local_addrlen = local_addrlen;

  if (client_ssl_init(c) != 0) {
    return -1;
  }

  if (client_quic_init(c, (struct sockaddr *)&remote_addr, remote_addrlen,
                       (struct sockaddr *)&local_addr, local_addrlen) != 0) {
    return -1;
  }

  c->stream.stream_id = -1;
  c->stream_id = -1;

  ev_io_init(&c->rev, read_cb, c->fd, EV_READ);
  c->rev.data = c;
  ev_io_start(EV_DEFAULT, &c->rev);

  ev_timer_init(&c->timer, timer_cb, 0., 0.);
  c->timer.data = c;
  ev_timer_again(EV_DEFAULT, &c->timer);

  ev_timer_init(&c->send_timer, send_timer_cb, 0., 0.05);
  c->send_timer.data = c;
  ev_timer_again(EV_DEFAULT, &c->send_timer);

  ev_io_init(&c->tun_watcher, read_tun_cb, c->tun_fd, EV_READ);
  c->tun_watcher.data = c;
  ev_io_start(EV_DEFAULT, &c->tun_watcher);

  return 0;
}

static void client_free(struct client *c) {
  ngtcp2_conn_del(c->conn);
  SSL_free(c->ssl);
  SSL_CTX_free(c->ssl_ctx);
}

void run_client_quicin(struct client c, void* fec_cfg, fec_callbacks* fec_cb) {
  if (fec_cb) {
    c.init_fec(fec_cfg, fec_cb);
  }
  
  c.tun_fd = tun_alloc("tun0", IFF_TUN | IFF_NO_PI);
  printf("device tun0 with fd %i created\n", c.tun_fd);

  srandom((unsigned int)timestamp());

  if (client_init(&c) != 0) {
    exit(EXIT_FAILURE);
  }

  if (client_write(&c) != 0) {
    exit(EXIT_FAILURE);
  }

  ev_run(EV_DEFAULT, 0);

  client_free(&c);
}

void log_time() {
  TimeLogger log_timer;
  log_timer.yield_time();
}

int main(int argc, char** argv) {
  CLI::App app{"QUIC sender"};

  std::string dst_ip, dst_port;
  app.add_option("dst_ip", dst_ip, "ip to forward data to")->required();
  app.add_option("dst_port", dst_port, "port to forward data to")->required();

  std::string dump_file = {};
  app.add_option("--dump_sent", dump_file, "file to write sent data to");

  bool quiet = false;
  app.add_flag("-q,--quiet", quiet, "disable logging");

  std::string cc;
  app.add_option("--cc", cc, "congestion control");

  std::string bbr2_params;
  app.add_option("--bbr2-params", bbr2_params, R"(<DOUBLE>,<DOUBLE>,<DOUBLE>,<DURATION>
  InOpSy parameters, defining constants in BBRv2:
    - BBRLossTresh: [0.0, 1.0]        default = 0.02
    - BBRBeta: [0.0, 1.0]             default = 0.7
    - BBRProbeRttCwndGain: [0.0, 1.0] default = 0.5
    - ProbeRTTDuration                default = 200 ms
  If parameters are not set, they equal default values. )"
  );

  double loss = 0.0;
  app.add_option("--loss", loss, "emulate loss when sending packets (single losses based on uniform distribution)");

  bool enable_fin_mode = false;
  app.add_flag("--fin", enable_fin_mode, "resend packets after TCP FIN without accumulating FEC chunks");

  bool inopsy_log_is_set = false;
  app.add_flag("--inopsy-log", inopsy_log_is_set, "use InOpSy type logs.");

  uint32_t num_packets, payload_size;
  auto& fec_1pr = *app.add_subcommand("fec_1pr");
  fec_1pr.add_option("num_packets", num_packets, "number of informational packets in batch")->required();
  fec_1pr.add_option("payload_size", payload_size, "size of payload to send in one packet")->required();

  auto& fec_2pr = *app.add_subcommand("fec_2pr");
  fec_2pr.add_option("num_packets", num_packets, "number of informational packets in batch")->required();
  fec_2pr.add_option("payload_size", payload_size, "size of payload to send in one packet")->required();

  auto& fec_r = *app.add_subcommand("fec_r");
  fec_r.add_option("num_packets", num_packets, "number of informational packets in batch")->required();
  fec_r.add_option("payload_size", payload_size, "size of payload to send in one packet")->required();

  auto& fec_rs = *app.add_subcommand("fec_rs");
  fec_rs.add_option("num_packets", num_packets, "number of informational packets in batch")->required();
  fec_rs.add_option("payload_size", payload_size, "size of payload to send in one packet")->required();

  uint32_t leo_num_recovery, leo_buf_size = 0;
  auto& fec_leo = *app.add_subcommand("fec_leo");
  fec_leo.add_option("num_packets", num_packets, "number of informational packets in batch")->required();
  fec_leo.add_option("payload_size", payload_size, "size of payload to send in one packet")->required();
  fec_leo.add_option("num_recovery", leo_num_recovery, "number of recovery packets in batch")->required();
  fec_leo.add_option("--buf_size", leo_buf_size, "size of leopard buffer, equal to payload size by default");

  auto& fec_dummy = *app.add_subcommand("fec_dummy");
  fec_dummy.add_option("num_packets", num_packets, "number of informational packets in batch")->required();
  fec_dummy.add_option("payload_size", payload_size, "size of payload to send in one packet")->required();

  CLI11_PARSE(app, argc, argv);

  ngtcp2_cc_algo cc_algo = NGTCP2_CC_ALGO_CUBIC;
  if (cc == "cubic") {
    cc_algo = NGTCP2_CC_ALGO_CUBIC;
  } else if (cc == "reno") {
    cc_algo = NGTCP2_CC_ALGO_RENO;
  } else if (cc == "bbr") {
    cc_algo = NGTCP2_CC_ALGO_BBR;
  } else if (cc == "bbr2") {
    cc_algo = NGTCP2_CC_ALGO_BBR2;
  } else if (!cc.empty()) {
    std::cerr << "invalid cc chosen: " << cc << "\nallowed options: cubic|reno|bbr|bbr2" << std::endl;
    exit(EXIT_FAILURE);
  };
  
  void* fec_cfg = nullptr;
  fec_callbacks* fec_cb = nullptr;
  if (fec_1pr) {
    fec_cfg = new FecXORCfg{payload_size, num_packets};
    fec_cb = &xor_callbacks;
  } else if (fec_2pr) {
    fec_cfg = new Fec2PRCfg{payload_size, num_packets};
    fec_cb = &pr2_callbacks;
  } else if (fec_r) {
    fec_cfg = new FecRCfg{payload_size, num_packets};
    fec_cb = &rscheme_callbacks;
  } else if (fec_rs) {
    fec_cfg = new FecRSCfg{payload_size, num_packets};
    fec_cb = &reedsol_callbacks;
  } else if (fec_leo) {
    if (!leo_buf_size) { leo_buf_size = payload_size; }
    fec_cfg = new FecLeoCfg{payload_size, num_packets, leo_num_recovery, leo_buf_size};
    fec_cb = &leopard_callbacks;
  } else if (fec_dummy) {
    fec_cfg = new FecDummyCfg{payload_size, num_packets};
    fec_cb = &dummy_callbacks;
  }

  double bbr2_loss_tresh = 0.02;
  double bbr2_beta = 0.7;
  double bbr2_probe_rtt_cwnd_gain = 0.5;
  double bbr2_probe_rtt_duration =  200 * NGTCP2_MILLISECONDS;
  if (!bbr2_params.empty()) {
    try {
      std::stringstream ss(bbr2_params);
      ss.exceptions(std::ios::failbit);
      std::string substr;
      getline(ss, substr, ',');
      bbr2_loss_tresh = std::stod(substr);
      getline(ss, substr, ',');
      bbr2_beta = std::stod(substr);
      getline(ss, substr, ',');
      bbr2_probe_rtt_cwnd_gain = std::stod(substr);
      getline(ss, substr);
      bbr2_probe_rtt_duration;
      if (auto t = ngtcp2::util::parse_duration(substr); !t) {
        throw std::runtime_error("--bbr2-params bbr2_probe_rtt_duration: invalid argument");
      } else {
        bbr2_probe_rtt_duration = *t;
      }
    } catch (const std::ifstream::failure& e) {
      throw std::runtime_error("failed to parse bbr2 parameters");
    }
  }

  
  if (inopsy_log_is_set) {
    quiet = true;
  }
  
  struct client c = {};
  c.inopsy_log_is_set = inopsy_log_is_set;
  c.dump_file = dump_file;
  c.disable_logging = quiet;
  c.loss = loss;
  c.enable_fin_mode = enable_fin_mode;
  c.cc_algo = cc_algo;
  c.bbr2_loss_tresh = bbr2_loss_tresh;
  c.bbr2_beta = bbr2_beta;
  c.bbr2_probe_rtt_cwnd_gain = bbr2_probe_rtt_cwnd_gain;
  c.bbr2_probe_rtt_duration = bbr2_probe_rtt_duration;
  strcpy(c.remote_host, dst_ip.c_str());
  strcpy(c.remote_port, dst_port.c_str());
  printf("remote: %s:%s\n", c.remote_host, c.remote_port);

  if (inopsy_log_is_set) {  
    std::thread log_timer_thread(log_time);
    std::thread client_thread(run_client_quicin, c, fec_cfg, fec_cb);

    std::cerr << "Point1\n";
    client_thread.join();
    std::cerr << "Point2\n";
  } else {
    run_client_quicin(c, fec_cfg, fec_cb);
  }

  return 0;
}
