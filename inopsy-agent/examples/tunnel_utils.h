#define SERVER_SCRIPT "server.sh"
#define CLIENT_SCRIPT "client.sh"

/**
 * Function to allocate a tunnel
 */
int tun_alloc(char *dev, int flags);

/**
 * Function to read from a tunnel
 */
int tun_read(int tun_fd, char *buffer, int length);

/**
 * Function to write to a tunnel
 */
int tun_write(int tun_fd, char *buffer, int length);
