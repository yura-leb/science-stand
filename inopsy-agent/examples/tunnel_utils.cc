#include "tunnel_utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <linux/if.h>
#include <linux/if_tun.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <netdb.h>
#include <pwd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cerrno>
#include <exception>

/**
 * Function to allocate a tunnel
 */
int tun_alloc(char *dev, int flags)
{
  struct ifreq ifr;
  int tun_fd, err;
  char *clonedev = "/dev/net/tun";
  printf("[DEBUG] Allocating tunnel\n");

  tun_fd = open(clonedev, O_RDWR);

  if(tun_fd == -1) {
    perror("Unable to open clone device\n");
    throw std::exception();
  }
  
  memset(&ifr, 0, sizeof(ifr));

  ifr.ifr_flags = flags;

  if (*dev) {
    strncpy(ifr.ifr_name, dev, IFNAMSIZ);
  }

  if ((err=ioctl(tun_fd, TUNSETIFF, (void *)&ifr)) < 0) {
    close(tun_fd);
    fprintf(stderr, "Error returned by ioctl(): %s\n", strerror(err));
    perror("Error in tun_alloc()\n");
    throw std::exception();
  }

  printf("[DEBUG] Created tunnel %s\n", dev);

  return tun_fd;
}

/**
 * Function to read from a tunnel
 */
int tun_read(int tun_fd, char *buffer, int length)
{
  int bytes_read;
  // printf("[DEBUG] Reading from tunnel\n");
  bytes_read = read(tun_fd, buffer, length);

  if (bytes_read == -1) {
    perror("Unable to read from tunnel\n");
    throw std::exception();
  }
  else {
    return bytes_read;
  }
}

/**
 * Function to write to a tunnel
 */
int tun_write(int tun_fd, char *buffer, int length)
{
  int bytes_written;
  // printf("[DEBUG] Writing to tunnel\n");
  bytes_written = write(tun_fd, buffer, length);

  if (bytes_written == -1) {
    printf("Unable to write to tunnel: %s\n", strerror(errno));
    throw std::exception();
  }
  else {
    return bytes_written;
  }
}
