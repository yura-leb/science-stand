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
#include "socket_utils.h"

// listen on IP1:PORT1
// send to IP2:PORT2
// supports only one connection, exits afterwards

int unsafe_main(int argc, char** argv) {
    CLI::App app{"Simplified split-TCP proxy"};

    std::string bind_ip, bind_port, dst_ip, dst_port;
    app.add_option("bind_ip", bind_ip, "IP for listening")->required();
    app.add_option("bind_port", bind_port, "port for listening")->required();
    app.add_option("dst_ip", dst_ip, "ip to forward data to")->required();
    app.add_option("dst_port", dst_port, "port to forward data to")->required();

    CLI11_PARSE(app, argc, argv);

    addrinfo *result, *rp;
    int ret, sock ;

    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    ret = getaddrinfo(bind_ip.c_str(), bind_port.c_str(), &hints, &result);
    for (rp = result; rp != nullptr; rp = rp->ai_next)
    {
        //  | SOCK_NONBLOCK
        sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock == -1) {
            continue;
        }

        if (bind(sock, rp->ai_addr, rp->ai_addrlen) == 0) {
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
    printf("Listening on %s:%s\n", bind_ip.c_str(), bind_port.c_str());

    auto send_sock = Socket{socket(AF_INET, SOCK_STREAM, 0)};
    auto send_sockfd = send_sock.fd;
    if (send_sockfd == -1) {
        printf("Failed to create sending socket: %s\n", strerror(errno));
        throw std::exception();
    }
    sockaddr_in servaddr = {};

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr(dst_ip.c_str());
    servaddr.sin_port = htons(std::stoi(dst_port));

    while (1) {
        if (connect(send_sockfd, (sockaddr*)&servaddr, sizeof(servaddr)) == 0) {
            printf("Connected to %s:%s\n", dst_ip.c_str(), dst_port.c_str());
            break;
        } else {
            printf("Failed to connect: %s\n", strerror(errno));
            sleep(5);
        }
    }

    auto connfd = accept(sock, nullptr, nullptr);
    if (connfd < 0) {
        printf("Failed to accept connection: %s\n", strerror(errno));
        throw std::exception();
    }
    printf("Accept connection\n");

    char buf[65536];
    while(1) {
        auto rcvd = recv(connfd, buf, sizeof(buf), 0);
        if (rcvd == -1) {
            printf("Failed to read data from socket: %s\n", strerror(errno));
            throw std::exception();
        }
        if (rcvd == 0) {
            printf("Shutdown connection\n");
            shutdown(send_sockfd, SHUT_WR);
            break;
        }

        auto sent = send(send_sockfd, buf, rcvd, 0);
        if (sent == -1) {
            printf("Failed to send data: %s\n", strerror(errno));
            throw std::exception();
        }

        if (rcvd != sent) {
          printf("Sent data amount differs from received\n");
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