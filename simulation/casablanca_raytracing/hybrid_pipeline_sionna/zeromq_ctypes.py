from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path

ZMQ_REQ = 3
ZMQ_REP = 4
ZMQ_LINGER = 17
ZMQ_RCVTIMEO = 27
ZMQ_SNDTIMEO = 28


class ZeroMqError(RuntimeError):
    pass


def _load_library() -> ctypes.CDLL:
    candidates = [
        ctypes.util.find_library("zmq"),
        "/lib64/libzmq.so.5",
        "/usr/lib64/libzmq.so.5",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            return ctypes.CDLL(str(path if path.exists() else candidate))
        except OSError:
            continue
    raise ZeroMqError("Unable to locate libzmq.so.5")


_lib = _load_library()
_lib.zmq_ctx_new.restype = ctypes.c_void_p
_lib.zmq_ctx_term.argtypes = [ctypes.c_void_p]
_lib.zmq_socket.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.zmq_socket.restype = ctypes.c_void_p
_lib.zmq_close.argtypes = [ctypes.c_void_p]
_lib.zmq_bind.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.zmq_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.zmq_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.zmq_send.restype = ctypes.c_int
_lib.zmq_recv.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.zmq_recv.restype = ctypes.c_int
_lib.zmq_setsockopt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lib.zmq_errno.restype = ctypes.c_int
_lib.zmq_strerror.argtypes = [ctypes.c_int]
_lib.zmq_strerror.restype = ctypes.c_char_p


def _raise_last_error(context: str) -> None:
    err = _lib.zmq_errno()
    message = _lib.zmq_strerror(err).decode("utf-8", errors="replace")
    raise ZeroMqError(f"{context}: {message}")


class ZmqContext:
    def __init__(self) -> None:
        self._ptr = _lib.zmq_ctx_new()
        if not self._ptr:
            _raise_last_error("zmq_ctx_new")

    def socket(self, socket_type: int) -> "ZmqSocket":
        return ZmqSocket(self, socket_type)

    def close(self) -> None:
        if self._ptr:
            rc = _lib.zmq_ctx_term(self._ptr)
            if rc != 0:
                _raise_last_error("zmq_ctx_term")
            self._ptr = None


class ZmqSocket:
    def __init__(self, context: ZmqContext, socket_type: int) -> None:
        self._ptr = _lib.zmq_socket(context._ptr, socket_type)
        if not self._ptr:
            _raise_last_error("zmq_socket")

    def set_int_option(self, option: int, value: int) -> None:
        int_value = ctypes.c_int(value)
        rc = _lib.zmq_setsockopt(self._ptr, option, ctypes.byref(int_value), ctypes.sizeof(int_value))
        if rc != 0:
            _raise_last_error("zmq_setsockopt")

    def bind(self, endpoint: str) -> None:
        rc = _lib.zmq_bind(self._ptr, endpoint.encode("utf-8"))
        if rc != 0:
            _raise_last_error(f"zmq_bind({endpoint})")

    def connect(self, endpoint: str) -> None:
        rc = _lib.zmq_connect(self._ptr, endpoint.encode("utf-8"))
        if rc != 0:
            _raise_last_error(f"zmq_connect({endpoint})")

    def send(self, payload: bytes) -> None:
        buffer = ctypes.create_string_buffer(payload)
        rc = _lib.zmq_send(self._ptr, buffer, len(payload), 0)
        if rc < 0:
            _raise_last_error("zmq_send")

    def recv(self, max_size: int = 1_048_576) -> bytes:
        buffer = ctypes.create_string_buffer(max_size)
        rc = _lib.zmq_recv(self._ptr, buffer, max_size, 0)
        if rc < 0:
            _raise_last_error("zmq_recv")
        return buffer.raw[:rc]

    def close(self) -> None:
        if self._ptr:
            rc = _lib.zmq_close(self._ptr)
            if rc != 0:
                _raise_last_error("zmq_close")
            self._ptr = None
