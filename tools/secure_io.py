"""No-follow access anchored to open directories (POSIX and Windows)."""

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import stat


def linked(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def windows_open(path, directory=False, create=False):
    """Open the entry itself; deny delete sharing until the handle closes."""
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    # Share read only: prevent rename/deletion and in-place reparse-point writes.
    handle = kernel.CreateFileW(str(path), 0x80 if directory else (0x40000000 if create else 0x80000000),
                                1, None, 1 if create else 3,
                                0x00200000 | (0x02000000 if directory else 0x80), None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    class AttributeTag(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("tag", wintypes.DWORD)]
    kernel.GetFileInformationByHandleEx.argtypes = (wintypes.HANDLE, ctypes.c_int,
                                                   wintypes.LPVOID, wintypes.DWORD)
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    try:
        info = AttributeTag()
        if not kernel.GetFileInformationByHandleEx(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if info.attributes & 0x400 or bool(info.attributes & 0x10) != directory:
            raise OSError("unexpected link or entry type")
        if directory:
            return handle, lambda: kernel.CloseHandle(handle)
        fd = msvcrt.open_osfhandle(handle, (os.O_WRONLY if create else os.O_RDONLY) | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    return fd, lambda: os.close(fd)


class Directory:
    def __init__(self, path, handle):
        self.path, self.handle = path, handle

    def entries(self):
        return os.scandir(self.path if os.name == "nt" else self.handle)

    @contextmanager
    def child(self, name, create=False):
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("expected a single directory entry")
        path = self.path / name
        if create:
            try:
                if os.name == "nt":
                    path.mkdir()
                else:
                    os.mkdir(name, dir_fd=self.handle)
            except FileExistsError:
                pass
        with open_directory(path, self, name) as directory:
            yield directory

    @contextmanager
    def file(self, name, create=False):
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("expected a single file entry")
        if os.name == "nt":
            fd, close = windows_open(self.path / name, create=create)
        else:
            flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL) if create else os.O_RDONLY
            fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=self.handle)
            close = lambda: os.close(fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("expected a regular file")
            stream = os.fdopen(fd, "w" if create else "rb", encoding="utf-8" if create else None)
        except BaseException:
            close()
            raise
        with stream:
            yield stream


@contextmanager
def open_directory(path, parent=None, name=None):
    if os.name == "nt":
        handle, close = windows_open(path, directory=True)
    else:
        handle = os.open(name if parent else path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                         dir_fd=parent.handle if parent else None)
        close = lambda: os.close(handle)
    try:
        yield Directory(path, handle)
    finally:
        close()


@contextmanager
def anchor(path, allow_missing=False):
    """Pin every existing component; defer missing output directories until write."""
    path = Path(os.path.abspath(path))
    with ExitStack() as stack:
        directory = stack.enter_context(open_directory(Path(path.anchor)))
        missing = []
        for name in path.parts[1:]:
            if missing:
                missing.append(name)
                continue
            try:
                directory = stack.enter_context(directory.child(name))
            except FileNotFoundError:
                if not allow_missing:
                    raise
                missing.append(name)
        yield directory, missing


@contextmanager
def materialize(directory, missing):
    with ExitStack() as stack:
        for name in missing:
            directory = stack.enter_context(directory.child(name, create=True))
        yield directory
