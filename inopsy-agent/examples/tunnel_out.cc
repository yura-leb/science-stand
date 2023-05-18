#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/ip.h>
#include <netdb.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <linux/if.h>
#include <linux/if_tun.h>

#include "CLI11.hpp"
#include "tunnel_utils.h"

int unsafe_main(int argc, char** argv) {
    CLI::App app{"Tunnel receiver endpoint"};

    std::string bind_ip, bind_port;
    app.add_option("bind_ip", bind_ip, "ip for listening")->required();
    app.add_option("bind_port", bind_port, "port for listening")->required();

    CLI11_PARSE(app, argc, argv);

    printf("start tunnel-out\n");

    auto tun_fd = tun_alloc("tun1", IFF_TUN | IFF_NO_PI);
    std::cout << "allocated tunnel with fd " << tun_fd << std::endl;

    addrinfo *result, *rp;
    int ret, sock;

    sockaddr local_addr; size_t local_addrlen;

    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    ret = getaddrinfo(bind_ip.c_str(), bind_port.c_str(), &hints, &result);
    for (rp = result; rp != nullptr; rp = rp->ai_next)
    {
        // TODO: | SOCK_NONBLOCK ?
        sock = socket(rp->ai_family, rp->ai_socktype,
            rp->ai_protocol);
        if (sock == -1) {
            continue;
        }

        if (bind(sock, rp->ai_addr, rp->ai_addrlen) == 0)
        {
            local_addrlen = rp->ai_addrlen;
            memcpy(&local_addr, rp->ai_addr, rp->ai_addrlen);
            break;
        }

        close(sock);
    }

    freeaddrinfo(result);

    if (rp == nullptr) {
        printf("Failed to create and bind socket: %s\n", strerror(errno));
        throw std::exception();
    }

    if (listen(sock, 5) == -1) {
        printf("Failed to start listening\n");
        throw std::exception();
    }

    auto connfd = accept(sock, nullptr, nullptr);
    if (connfd < 0) {
        printf("Failed to accept connection: %s\n", strerror(errno));
        throw std::exception();
    } 

    char buf[65536];
    while(1) {
        auto datalen = recv(connfd, buf, sizeof(buf), 0);
        if (datalen == -1) {
            printf("Failed to read data from socket: %s\n", strerror(errno));
            throw std::exception();
        }
        // uncomment for debugging data transfer
        // printf("rcvd: %li", datalen);
        // TODO: if rcvd 0, return

        tun_write(tun_fd, (char*)buf, datalen);
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