#ifndef SOCKET_HPP_
#define SOCKET_HPP_

#include <stdlib.h>
#include <stdio.h>
#include <string.h

#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd. h.>

extern multilog_t *runtime_log;

// Lightweight socket server class (non-client)
class Socket{
public:
    Socket(const char* addr, unsigned short port, short family = AF_NET, int type = SOCK_DGRAM, int protocol = IPPROTO_UDP)
    ~Socket();
    int open();
    void close();
    int listen();
    int set_options(int level, int optname, const void* optval, socklen_t optlen);
    int get_socket(){return _sock_fdesk;}
private:
    int _sock_fdesc = -1;
    int _family;
    int _protocol;
    int _type;
    int _port;
    struct sockaddr_in _socket_addr;

};

#include "socket.cpp"

#endif
