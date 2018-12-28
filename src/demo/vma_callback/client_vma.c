#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <msgheader.h>
#include <mellanox/vma_extra.h>

#define PATTERN_SZ	64

char*
create_buf(int size)
{
	char	*buf = (char *)malloc(size);
	int		c = 'A', i;

	if (buf == NULL) {
		return NULL;
	}

	/*
	 * Fill buff with certain pattern
	 */
	for (i = 0; (i + PATTERN_SZ) < size;  i+= PATTERN_SZ) {

		if (c == 'Z') {
			c = 'A';
		}
		memset(buf, c, PATTERN_SZ);
		c++;
	}
	if (i < size) {
		memset(buf, c, (size - i));
	}
	return buf;
}

int
main(int argc, char *argv[])
{
	int sockfd, portno, n;
	struct sockaddr_in serv_addr;
	struct hostent *server;
	int	bufcount, bufsize;
	char *buffer;
  	int i, sleeptime;
	
	if (argc < 6) {
		fprintf(stderr,"usage %s <hostname> "
			"<port> <bufsize> <bufcount> <sleepinmicro>\n", argv[0]);
		exit(0);
	}
	
	portno = atoi(argv[2]);
	bufsize = atoi(argv[3]);
	bufcount = atoi(argv[4]);
	sleeptime = atoi(argv[5]);
	
	/*
	 * Create a socket point
	 */
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	
	if (sockfd < 0) {
		perror("ERROR opening socket");
		exit(1);
	}
	
	server = gethostbyname(argv[1]);
	
	if (server == NULL) {
		fprintf(stderr,"ERROR, no such host\n");
		exit(0);
	}
	
	bzero((char *) &serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	bcopy((char *)server->h_addr,
		(char *)&serv_addr.sin_addr.s_addr, server->h_length);
	serv_addr.sin_port = htons(portno);
	
	/*
	 * Now connect to the server
	 */
	if (connect(sockfd, (struct sockaddr*)&serv_addr,
				sizeof(serv_addr)) < 0) {
		perror("ERROR connecting");
		exit(1);
	}
	
	/*
	 * Now ask for a message from the user, this message
 	 * will be read by server
	 */
	int sz = 0;
	int flag = 1;
	int pid = getpid();
	msgheader_t	header;
	struct iovec	iov[2];
	struct vma_api_t	*vma_api = NULL;

	int result = setsockopt(sockfd,      /* socket affected */
                        IPPROTO_TCP,     /* set option at TCP level */
                        TCP_NODELAY,     /* name of option */
                        (char *) &flag,  /* the cast is historical cruft */
                        sizeof(int));    /* length of option value */
	if (result < 0) {
		perror("Error in disabling nagle algo");
	}

	vma_api = vma_get_api();
	if (vma_api == NULL) {
		printf("Failed to init vma_api\n");
		return -1;
	}

	buffer = create_buf(bufsize - sizeof(msgheader_t));
	if (buffer == NULL) {
		perror("Failed to create buffer\n");
		exit (1);
	}

	for (i = 0; i < bufcount; i++) { 

		/*
		 * Prepare & set header & data in iovec
		 */
		header.pid = pid;
		header.seqno = i;
		header.length = bufsize - sizeof (msgheader_t);
		iov[0].iov_base = &header;
		iov[0].iov_len = sizeof (msgheader_t);
		iov[1].iov_base = buffer;
		iov[1].iov_len = bufsize - sizeof(msgheader_t);

		n = writev(sockfd, iov, 2);
		if (n < 0 || n != bufsize) {
			perror("ERROR writing to socket");
			exit(1);
		} else {
			sz += bufsize;
			printf("Buf[%d] sent (Total data sent: %d)\n", i, sz);
			usleep(sleeptime);
		}
	}
	printf("##Overall Total data sent = %d bytes \n", sz);

	shutdown(sockfd, 1);
	return 0;
}
