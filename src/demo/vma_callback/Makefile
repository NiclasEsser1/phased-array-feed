CC := gcc
CFLAGS := -g -I.

all: server_vma_cb server_vma_zcopy client_vma

server_vma_cb: server_vma_cb.c
	$(CC) -I /usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include  -lglib-2.0  $(CFLAGS) -o $@ $<

server_vma_zcopy: server_vma_zcopy.c
	$(CC) $(CFLAGS) -o $@ $<

client_vma: client_vma.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f server_vma_zcopy client_vma server_vma_cb
