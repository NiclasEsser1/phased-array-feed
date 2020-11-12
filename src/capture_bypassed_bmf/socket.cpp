#ifdef SOCKET_HPP_

Socket::Socket(const char* addr, unsigned short port, short family, int type, int protocol)
    : _family(family), _type(type), _protocol(protocol), _port(htons(port))
{
    _socket_addr.sin_port = _port;
    _socket_addr.sin_family = _family;
    inet_pton(_socket_addr.sin_family, addr, &_socket_addr.sin_addr);
}

Socket::~Socket()
{
    this->close();
}

int Socket::open()
{
    _sock_fdesc = socket(_socket_addr.sin_family, _type, _protocol);
    if(_sock_fdesk < 0)
    {

    }
}
#endif
