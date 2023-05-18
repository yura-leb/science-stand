#include <iostream>
#include <netdb.h>
#include <memory.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <errno.h>
#include <linux/if_tun.h>
#include <netinet/tcp.h>

#include "CLI11.hpp"
#include "socket_utils.h"
#include "tunnel_utils.h"

// value estimated from tcpdump output on truncated IP packets
// depends on MTU of interface tun0
#define READLEN 1400

int unsafe_main(int argc, char** argv) {
    CLI::App app{"Tunnel sender endpoint"};

    std::string dst_ip, dst_port;
    app.add_option("dst_ip", dst_ip, "ip of receiver endpoint")->required();
    app.add_option("dst_port", dst_port, "port of receiver endpoint")->required();

    CLI11_PARSE(app, argc, argv);

    printf("start tunnel-in\n");
    auto tun_fd = tun_alloc("tun0", IFF_TUN | IFF_NO_PI);

    auto sock = Socket{socket(AF_INET, SOCK_STREAM, 0)};
    auto sockfd = sock.fd;
    if (sockfd == -1) {
        printf("Failed to create socket: %s\n", strerror(errno));
        throw std::exception();
    }
    sockaddr_in servaddr = {};

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr(dst_ip.c_str());
    servaddr.sin_port = htons(std::stoi(dst_port));

    while (1) {
        if (connect(sockfd, (sockaddr*)&servaddr, sizeof(servaddr)) == 0) {
            break;
        } else {
            printf("Failed to connect: %s\n", strerror(errno));
            sleep(5);
        }
    }

    char yes = 1;
    int res = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *) &yes, sizeof(int));
    if (res < 0) {
        printf("Failed to set TCP_NODELAY: %s\n", strerror(errno));
        throw std::exception();
    }

    char buf[READLEN];
    while (1) {
        auto rcvd = tun_read(tun_fd, buf, sizeof(buf));
        auto sent = send(sockfd, buf, rcvd, 0);
        // uncomment for debugging data transfer
        // printf("rcvd %li, sent %li", rcvd, sent);
        // TODO: in general probably need while {sent} in case sent < rcvd
        if (sent < 0) {
            printf("error sending: %s\n", strerror(errno));
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    try {
        return unsafe_main(argc, argv);
    } catch (const std::exception&) {
        return EXIT_FAILURE;
    }
}