#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <errno.h>
#include <mellanox/vma_extra.h>
#include <glib.h>

#define MAXEVENTS 64
int	total_iovec_sz;

/*
 * Define pthread function, which processes VM packets and then frees it.
 * Pthread based signalling
 */
typedef struct vma_packet_t vma_packet_t;
typedef struct vma_packets_t vma_packets_t;
struct vma_api_t	*vma_api;

typedef vma_recv_callback_retval_t
		(*vma_recv_callback_t)(int fd, size_t sz_iov,
			struct iovec iov[], struct vma_info_t* vma_info,
			void *context);
/*
vma_recv_callback_retval_t
server_vma_recv_pkt_notify_callback(int fd,
				size_t iov_sz,
				struct iovec iov[],
				struct vma_info_t* vma_info,
				void *context); */

vma_recv_callback_retval_t
server_vma_recv_pkt_notify_callback(int fd,
				size_t iov_sz,
				struct iovec iov[],
				struct vma_info_t* vma_info,
				void *context)
{
	int	i;
	int	curr_iovec_sz = 0;

	for (i = 0; i < iov_sz; i++) {
		curr_iovec_sz += iov[i].iov_len;
	}
	g_atomic_int_add(&total_iovec_sz, curr_iovec_sz);
	
	printf("Callback: Recieved VMA packet %p containing data of length: %d, Total data recieved until now: %d\n",
			vma_info->packet_id, curr_iovec_sz, g_atomic_int_get(&total_iovec_sz));
	return VMA_PACKET_RECV;
}


static int
make_socket_non_blocking (int sfd)
{
	int flags, s;

	flags = fcntl (sfd, F_GETFL, 0);
	if (flags == -1)
	{
		perror ("fcntl");
		return -1;
	}

	flags |= O_NONBLOCK;
	s = fcntl (sfd, F_SETFL, flags);
	if (s == -1)
	{
		perror ("fcntl");
		return -1;
	}

	return 0;
}

static int
create_and_bind (char *addr, char *port)
{
	struct addrinfo hints;
	struct addrinfo *result, *rp;
	int s, sfd;

	memset (&hints, 0, sizeof (struct addrinfo));

	hints.ai_family = AF_UNSPEC;	 /* Return IPv4 and IPv6 choices */
	hints.ai_socktype = SOCK_STREAM; /* We want a TCP socket */
	hints.ai_flags = AI_PASSIVE;	 /* All interfaces */

	s = getaddrinfo (addr, port, &hints, &result);
	if (s != 0)
	{
		fprintf (stderr, "getaddrinfo: %s\n", gai_strerror (s));
		return -1;
	}

	for (rp = result; rp != NULL; rp = rp->ai_next)
	{
		sfd = socket (rp->ai_family, rp->ai_socktype, rp->ai_protocol);
		if (sfd == -1)
			continue;

		s = bind (sfd, rp->ai_addr, rp->ai_addrlen);
		if (s == 0)
		{
			/*
			 * We managed to bind successfully!
			 */
			break;
		}
		close (sfd);
	}

	if (rp == NULL)
	{
		fprintf (stderr, "Could not bind\n");
		return -1;
	}

	freeaddrinfo (result);

	return sfd;
}

/*
 * It only finds the total data length in vm_packet segments
 * by traversing iovec
 */
int
process_packet_func(vma_packet_t *vma_packet)
{
	int packet_datalen = 0;
	int	k;

	for ( k =0; k < vma_packet->sz_iov;k++) {
		packet_datalen += vma_packet->iov[k].iov_len;
	}
	return packet_datalen;
}
	
int
do_read(int fd, struct vma_api_t *vma_api)
{
	ssize_t count;
	char *buf = NULL;
	int flags, j, k;
        int ret;
	vma_packets_t *vma_packets;
	int	datalen = 0;
        ssize_t sz = 8096 * 4096;
 	buf = (char *)malloc(sz);
	static int total_iovec_datalen = 0;
	static int total_ret_datalen = 0;

	flags = 0;
	retry:
	ret = vma_api->recvfrom_zcopy(fd, buf, sz, 
        			&flags, NULL, NULL);
	if (ret > 0) {
		if ((flags & MSG_VMA_ZCOPY) == MSG_VMA_ZCOPY) {
			vma_packets = (vma_packets_t*)buf;

			total_ret_datalen += ret;
			for (j = 0; j < vma_packets->n_packet_num; j++) {
				vma_packet_t* vma_packet = &vma_packets->pkts[j];

				datalen += process_packet_func(vma_packet);
				printf("recvfrom_zcopy: Packet_id: %p datalen: %d\n", vma_packet->packet_id, datalen);
			}

			if (datalen != ret) {
	//			fprintf(stderr, "ERROR: Datalen from vma_packet's iovec: %d and "
	//				"Return value from recvfrom_zcopy: %d do not Match\n",
	//				datalen, ret);
				goto retry;
			}
			total_iovec_datalen += datalen;

		} else {
			printf("Received non-zero-copy buf of size %d", ret);
		}
		if (vma_api->free_packets(fd, vma_packets->pkts, vma_packets->n_packet_num) < 0) {
				   perror("Error: ");
		}
	}
	//printf("Recvfrom_zcopy: Total data on socketfd: %d = Based on iovec: %d bytes, "
	//		"Based on return val: %d bytes\n", fd, total_iovec_datalen, total_ret_datalen);
	free(buf);
}

int
accept_connection(int sfd)
{
	struct sockaddr in_addr;
	socklen_t in_len;
	int infd = -1;
	char hbuf[NI_MAXHOST], sbuf[NI_MAXSERV];

	in_len = sizeof in_addr;
	infd = accept (sfd, &in_addr, &in_len);
	if (infd == -1)
	{
		if ((errno == EAGAIN) ||
			(errno == EWOULDBLOCK))
		{
			/*
			 * We have processed all incoming
			 * connections
			 */
			return -2;
		} else {
			perror ("accept");
			return -1;
		}
	}

	if (getnameinfo (&in_addr, in_len,
			 hbuf, sizeof hbuf,
			 sbuf, sizeof sbuf,
		 NI_NUMERICHOST | NI_NUMERICSERV) == 0) {

		printf("Accepted connection on descriptor %d "
			"(host=%s, port=%s)\n", infd, hbuf, sbuf);
	}

	/*
	 * Make the incoming socket non-blocking and add it to the
	 * list of fds to monitor.
	 */
	if (make_socket_non_blocking (infd) < 0) {
		perror("make_socket_non_blocking");
		return -1;
	}
	return infd;
}

int
main(int argc, char *argv[])
{
	int sfd = -1, s;
	int efd;
	struct epoll_event event;
	struct epoll_event *events;
	
	int	ret = 0;

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <address> <port>\n", argv[0]);
		return -1;
	}

	sfd = create_and_bind(argv[1], argv[2]);
	if (sfd == -1) {
		perror("create_and_bind");
		return -1;
	}

	if (make_socket_non_blocking(sfd) < 0) {
		perror("make_socket_non_blocking");
		ret = -1;
		goto cleanup;
	}

	if (listen(sfd, SOMAXCONN) < 0) {
		perror ("listen");
		ret = -1;
		goto cleanup;
	}

	vma_api = vma_get_api();
	if (vma_api == NULL) {
		perror("vma_get_api");
		ret = -1;
		goto cleanup;
	}


	/*if (vma_api->register_recv_callback(sfd,
			server_vma_recv_pkt_notify_callback,
			&total_iovec_sz) != 0) {
		printf("Failed to register callback");
		ret = -1;
		goto cleanup;
	}
	printf("callback registered\n"); */

	efd = epoll_create1(0);
	if (efd == -1)
	{
		perror ("epoll_create");
		ret = -1;
		goto cleanup;
	}

	event.data.fd = sfd;
	event.events = EPOLLIN | EPOLLET;
	if (epoll_ctl(efd, EPOLL_CTL_ADD, sfd, &event) < 0) {
		perror ("epoll_ctl: failed to set EPOLLET");
		ret = -1;
		goto cleanup;
	}

	/*
	 * Buffer where events are returned
	 */
	events = calloc(MAXEVENTS, sizeof (event));
	if (events == NULL) {
		perror("Memory alloc");
		ret = -1;
		goto cleanup;
	}


	/*
	 * The event loop
	 */
	while (1)
	{
		int n, i;

		n = epoll_wait(efd, events, MAXEVENTS, -1);
		for (i = 0; i < n; i++)
		{
			if ((events[i].events & EPOLLERR) ||
					(events[i].events & EPOLLHUP) ||
					(!(events[i].events & EPOLLIN)))
			{
				/*
				 * An error has occured on this fd, or the socket is not
				 * ready for reading (why were we notified then?)
				 */
				fprintf (stderr, "epoll error\n");
				close (events[i].data.fd);
				continue;
			} else if (sfd == events[i].data.fd) {
				/*
				 * We have a notification on the listening socket, which
				 * means one or more incoming connections.
				 */
				while (1)
				{
					int infd;

					infd = accept_connection(sfd);
					if (infd < 0) {
						break;
					}

					if (vma_api->register_recv_callback(infd,
						server_vma_recv_pkt_notify_callback,
						&total_iovec_sz) != 0) {

						printf("Failed to register callback");
						ret = -1;
						goto cleanup;
					}
					printf("callback registered\n");


					event.data.fd = infd;
					event.events = EPOLLIN | EPOLLET;
					if (epoll_ctl (efd, EPOLL_CTL_ADD, infd, &event) < 0) {
						perror ("epoll_ctl");
						goto cleanup;
					}
				}
				continue;
			} else {
				/*
				 * We have data on the fd waiting to be read. Read it
				 */

				if (do_read(events[i].data.fd, vma_api) < 0) {
					fprintf(stderr, "do_read failed\n");
				}
			}
		}
	}

cleanup:
	if (events) {
		free (events);
	}

	if (sfd != -1) {
		close (sfd);
	}

	return EXIT_SUCCESS;
}
