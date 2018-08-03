#ifndef _SYNC_H
#define _SYNC_H

#include "capture.h"

int threads(conf_t *conf);
void *sync_thread(void *conf);

#endif
