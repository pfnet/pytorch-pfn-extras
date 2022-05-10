import collections
import os
import sys
import time
from typing import Deque, Optional, Sequence, TextIO, Tuple

from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


try:
    from IPython import get_ipython
    _ipython_available = True
except ImportError:
    _ipython_available = False


def _is_notebook() -> bool:
    if _ipython_available and get_ipython() is not None:
        return 'IPKernelApp' in get_ipython().config
    return False


if os.name == 'nt':
    import ctypes
    from ctypes import windll  # type: ignore [attr-defined]

    _STD_OUTPUT_HANDLE = -11

    _COORD = ctypes.wintypes._COORD

    class _CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
        _fields_ = [('dwSize', _COORD), ('dwCursorPosition', _COORD),
                    ('wAttributes', ctypes.c_ushort),
                    ('srWindow', ctypes.wintypes.SMALL_RECT),
                    ('dwMaximumWindowSize', _COORD)]

    def set_console_cursor_position(x: int, y: int) -> None:
        """Set relative cursor position from current position to (x,y)"""

        whnd = windll.kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
        csbi = _CONSOLE_SCREEN_BUFFER_INFO()
        windll.kernel32.GetConsoleScreenBufferInfo(whnd, ctypes.byref(csbi))
        cur_pos = csbi.dwCursorPosition
        pos = _COORD(cur_pos.X + x, cur_pos.Y + y)

        # Workaround the issue that pyreadline overwrites the argtype
        setpos = windll.kernel32.SetConsoleCursorPosition
        argtypes = setpos.argtypes
        setpos.argtypes = None
        setpos(whnd, pos)
        setpos.argtypes = argtypes

    def erase_console(x: int, y: int, mode: int = 0) -> None:
        """Erase screen.

        Mode=0: From (x,y) position down to the bottom of the screen.
        Mode=1: From (x,y) position down to the beginning of line.
        Mode=2: Hole screen
        """

        whnd = windll.kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
        csbi = _CONSOLE_SCREEN_BUFFER_INFO()
        windll.kernel32.GetConsoleScreenBufferInfo(whnd, ctypes.byref(csbi))
        cur_pos = csbi.dwCursorPosition
        wr = ctypes.c_ulong()
        if mode == 0:
            num = csbi.srWindow.Right * (
                csbi.srWindow.Bottom - cur_pos.Y) - cur_pos.X
            windll.kernel32.FillConsoleOutputCharacterA(
                whnd, ord(' '), num, cur_pos, ctypes.byref(wr))
        elif mode == 1:
            num = cur_pos.X
            windll.kernel32.FillConsoleOutputCharacterA(
                whnd, ord(' '), num, _COORD(0, cur_pos.Y), ctypes.byref(wr))
        elif mode == 2:
            os.system('cls')


class ProgressBar:

    def __init__(self, out: Optional[TextIO] = None) -> None:
        self._out = sys.stdout if out is None else out
        self._recent_timing: Deque[Tuple[int, float, float]] = collections.deque(
            [], maxlen=100)

    def update_speed(
            self,
            iteration: int,
            epoch_detail: float
    ) -> Tuple[float, float]:
        now = time.time()
        self._recent_timing.append((iteration, epoch_detail, now))
        old_t, old_e, old_sec = self._recent_timing[0]
        span = now - old_sec
        if span != 0:
            speed_t = (iteration - old_t) / span
            speed_e = (epoch_detail - old_e) / span
        else:
            speed_t = float('inf')
            speed_e = float('inf')
        return speed_t, speed_e

    def get_lines(self) -> Sequence[str]:
        raise NotImplementedError

    def update(
            self,
            manager: Optional[ExtensionsManagerProtocol] = None
    ) -> None:
        self.erase_console()

        lines = self.get_lines()
        for line in lines:
            self._out.write(line)

        self.move_cursor_up(len(lines))
        self.flush()

    def close(self) -> None:
        self.erase_console()
        self.flush()

    def erase_console(self) -> None:
        if os.name == 'nt':
            erase_console(0, 0)
        else:
            self._out.write('\033[J')

    def move_cursor_up(self, n: int) -> None:
        # move the cursor to the head of the progress bar
        if os.name == 'nt':
            set_console_cursor_position(0, - n)
        else:
            self._out.write('\033[{:d}A'.format(n))

    def flush(self) -> None:
        if hasattr(self._out, 'flush'):
            self._out.flush()
