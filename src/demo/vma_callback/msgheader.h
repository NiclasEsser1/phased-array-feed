#ifndef _MSGHEADER_H_
#define _MSGHEADER_H_

//#define CTLR_MSG	0x00000000	/* Messages expected from client */
//#define	CTLR_ACK	0x00000001	/* Ack to messages recieved from client */
//#define CTLR_ERR	0x00000002	/* Error in recieving packet, payload contain error packet info */
//
//typedef struct cltrheader {
//	char type;			/* Msg or Ack or Error */
//	int length;			/* Length of payload */
//	char payload[0];	/* Payload */
//} ctlrheader_t;
//
//typedef struct cltrmsg {
//	int	magicno;		/* Unique across one communication */
//	int	start_seqno;	/* Message with start sequence num */
//	int	end_seqno;		/* Message with end sequence num */
//} cltrmsg_t;
//
//*typedef struct ctlrack {
//	int magicno;		/* Acknowldgement for magic number */
//	int 
//	
//typedef struct cltrack {
//} ctlrack_t;
//
typedef struct msgheader {
	int	pid;		/* Pid of the process */
	int	seqno;		/* Process generated unique seq no */
	int	length;		/* Length of the message */
} msgheader_t;

#endif /* _MSGHEADER_H_ */
