"""Standard-library remote socket lease, sent to the jump host over SSH.

Hold advisory locks for the life of one SSH master. Existing live sockets are
never replaced. Stale sockets may be removed only while the lock is held.
The BOUND handshake records inodes after SSH has created all requested sockets;
cleanup removes only those same inodes. No TCP listener is created here.
"""

import errno
import fcntl
import os
from pathlib import Path
import socket
import stat
import sys


READY = "PLACEHOLDER_RELAY_READY"
BOUND = "PLACEHOLDER_RELAY_BOUND"


def remove_stale(path):
    try:
        original = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(original.st_mode) or original.st_uid != os.getuid():
        raise ValueError(f"Refusing an unowned/non-socket path: {path}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        result = probe.connect_ex(str(path))
    if result != errno.ECONNREFUSED:
        raise ValueError(f"Socket is live or cannot be proven stale: {path} (connect={result})")
    current = path.lstat()
    if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
        raise ValueError(f"Socket changed while checking staleness: {path}")
    path.unlink()


def lease(paths):
    locks, owned = [], {}
    try:
        for path in sorted(paths):
            parent = path.parent
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = parent.lstat()
            if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o077):
                raise ValueError(f"Socket directory must be owned by this user with mode 700: {parent}")
            lock = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            locks.append(lock)
            metadata = os.fstat(lock)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
                raise ValueError(f"Socket lock must be owned by this user with mode 600: {path}.lock")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in paths:
            remove_stale(path)
        print(READY, flush=True)
        message = sys.stdin.readline()
        if message != "BOUND\n":
            return 0
        for path in paths:
            metadata = path.lstat()
            if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise ValueError(f"Expected owned SSH socket after forwarding: {path}")
            owned[path] = (metadata.st_dev, metadata.st_ino)
            path.chmod(0o600)
        print(BOUND, flush=True)
        while sys.stdin.buffer.read(4096):
            pass
        return 0
    finally:
        for path, identity in owned.items():
            try:
                metadata = path.lstat()
                if (metadata.st_dev, metadata.st_ino) == identity:
                    path.unlink()
            except FileNotFoundError:
                pass
        for lock in reversed(locks):
            os.close(lock)


def main():
    try:
        return lease([Path(value) for value in sys.argv[1:]])
    except (OSError, ValueError) as exc:
        print(f"[relay socket lease] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
