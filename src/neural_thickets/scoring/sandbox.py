"""Linux-only resource isolation for executing untrusted MBPP generations.

This is defense in depth, not a general-purpose security boundary. The scorer
requires Linux Landlock, seccomp, root-to-nobody privilege dropping, and
process-group cleanup. Use a disposable container or VM as the outer security
boundary.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import resource
import select
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def isolation_available() -> tuple[bool, str]:
    if sys.platform != "linux":
        return False, "secure MBPP scoring requires Linux"
    if os.geteuid() != 0:
        return False, "secure MBPP scoring requires root to drop to uid/gid 65534"
    try:
        ctypes.CDLL("libseccomp.so.2")
    except OSError:
        return False, "libseccomp.so.2 is unavailable"
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(444, 0, 0, 1) < 3:
        return False, "Landlock ABI >= 3 is unavailable"
    return True, "available"


def isolated_call(function: Callable[[], Any], *, timeout_seconds: int = 30) -> dict[str, Any]:
    available, reason = isolation_available()
    if not available:
        raise RuntimeError(
            f"{reason}. Run scoring in a disposable, capability-limited Linux container; "
            "subprocesses alone are not a security sandbox."
        )
    with tempfile.TemporaryDirectory(prefix="neural_thickets_mbpp_", dir="/tmp") as scratch:
        os.chown(scratch, 65534, 65534)
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            _child(function, scratch, read_fd, write_fd)
        os.close(write_fd)
        deadline = time.monotonic() + timeout_seconds
        chunks = []
        total = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.killpg(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                os.close(read_fd)
                return {"ok": False, "status": "timeout", "reason": "sandbox wall timeout"}
            readable, _, _ = select.select([read_fd], [], [], remaining)
            if not readable:
                continue
            chunk = os.read(read_fd, 65_536)
            if not chunk:
                break
            total += len(chunk)
            if total > 1 << 20:
                os.killpg(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                os.close(read_fd)
                return {
                    "ok": False,
                    "status": "process_failure",
                    "reason": "sandbox result exceeded 1 MiB",
                }
            chunks.append(chunk)
        os.close(read_fd)
        data = b"".join(chunks)
        _, status = os.waitpid(pid, 0)
        if not data:
            return {
                "ok": False,
                "status": "process_failure",
                "reason": f"sandbox child exit status {status}",
            }
        return json.loads(data)


def _child(
    function: Callable[[], Any],
    scratch: str,
    read_fd: int,
    write_fd: int,
) -> None:
    os.close(read_fd)
    os.setsid()
    try:
        null_fd = os.open("/dev/null", os.O_RDWR)
        for descriptor in (0, 1, 2):
            os.dup2(null_fd, descriptor)
        if null_fd != write_fd:
            os.close(null_fd)
        os.closerange(3, write_fd)
        os.closerange(write_fd + 1, 1_048_576)
        os.environ.clear()
        os.environ.update(PATH="/usr/bin:/bin", HOME=scratch, TMPDIR=scratch)
        os.chdir(scratch)
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
        resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        _restrict(scratch)
        result = {"ok": True, "result": function()}
    except BaseException as error:
        result = {
            "ok": False,
            "status": "exception",
            "reason": f"{type(error).__name__}: {str(error)[:500]}",
        }
    payload = json.dumps(result).encode()
    if len(payload) > 1 << 20:
        payload = json.dumps(
            {
                "ok": False,
                "status": "process_failure",
                "reason": "sandbox result exceeded 1 MiB",
            }
        ).encode()
    view = memoryview(payload)
    while view:
        written = os.write(write_fd, view)
        view = view[written:]
    os.close(write_fd)
    os._exit(0)


def _restrict(scratch: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    seccomp = ctypes.CDLL("libseccomp.so.2")

    class Ruleset(ctypes.Structure):
        _fields_ = [("fs", ctypes.c_uint64)]

    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("access", ctypes.c_uint64), ("fd", ctypes.c_int32)]

    rights = (1 << 15) - 1
    ruleset = Ruleset(rights)
    ruleset_fd = libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if ruleset_fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, "landlock_create_ruleset failed")
    roots = {
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path("/usr/lib"),
        Path("/lib"),
        Path("/lib64"),
        Path(scratch),
    }
    for path in sorted(roots):
        if not path.exists():
            continue
        parent_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        allowed = rights if path == Path(scratch) else (1 | 4 | 8)
        rule = Rule(allowed, parent_fd)
        result = libc.syscall(445, ruleset_fd, 1, ctypes.byref(rule), 0)
        os.close(parent_fd)
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, f"landlock_add_rule failed for {path}")
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "PR_SET_NO_NEW_PRIVS failed")
    if libc.syscall(446, ruleset_fd, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "landlock_restrict_self failed")
    os.close(ruleset_fd)

    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("seccomp_init failed")
    denied = (
        "socket",
        "socketpair",
        "connect",
        "bind",
        "listen",
        "accept",
        "accept4",
        "clone",
        "clone3",
        "fork",
        "vfork",
        "execve",
        "execveat",
        "ptrace",
        "kill",
        "tkill",
        "tgkill",
        "process_vm_readv",
        "process_vm_writev",
        "mount",
        "umount2",
        "unshare",
        "setns",
        "chroot",
        "bpf",
        "perf_event_open",
        "userfaultfd",
        "open_by_handle_at",
        "io_uring_setup",
        "io_uring_enter",
        "io_uring_register",
        "reboot",
    )
    for name in denied:
        number = seccomp.seccomp_syscall_resolve_name(name.encode())
        if number < 0:
            seccomp.seccomp_release(context)
            raise RuntimeError(f"seccomp could not resolve syscall {name}")
        result = seccomp.seccomp_rule_add(
            context,
            0x00050000 | errno.EPERM,
            number,
            0,
        )
        if result != 0:
            seccomp.seccomp_release(context)
            raise OSError(-result, f"seccomp_rule_add failed for {name}")
    result = seccomp.seccomp_load(context)
    seccomp.seccomp_release(context)
    if result != 0:
        raise OSError(-result, "seccomp_load failed")

