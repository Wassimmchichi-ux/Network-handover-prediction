#ifndef MINI_ZMQ_H
#define MINI_ZMQ_H

#include <cstddef>

#define ZMQ_REQ 3
#define ZMQ_REP 4
#define ZMQ_LINGER 17
#define ZMQ_RCVTIMEO 27
#define ZMQ_SNDTIMEO 28

extern "C" {
void* zmq_ctx_new(void);
int zmq_ctx_term(void* context);
void* zmq_socket(void* context, int type);
int zmq_close(void* socket);
int zmq_bind(void* socket, const char* endpoint);
int zmq_connect(void* socket, const char* endpoint);
int zmq_send(void* socket, const void* buf, size_t len, int flags);
int zmq_recv(void* socket, void* buf, size_t len, int flags);
int zmq_setsockopt(void* socket, int option_name, const void* option_value, size_t option_len);
int zmq_errno(void);
const char* zmq_strerror(int errnum);
}

#endif
