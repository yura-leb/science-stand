#include <unistd.h>

/**
 * @brief Socket wrapper with destructor
 */
class Socket {
public:
    const int fd;

    Socket(int fd) : fd(fd) {}

    Socket(Socket &&) = default;
    Socket(const Socket&) = delete;
    Socket &operator=(const Socket&) = delete;
    Socket &operator=(Socket &&) = delete;

    ~Socket() {
        close(fd);
    }
};
