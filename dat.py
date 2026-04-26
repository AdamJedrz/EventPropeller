import os
import sys
from typing import Dict, Iterator, Tuple

import numpy as np

EV_TYPES = {
    0: [("t", "u4"), ("_", "i4")],
    12: [("t", "u4"), ("_", "i4")],
    14: [("p", "i2"), ("t", "i8"), ("id", "i2")],
    40: [("t", "u4"), ("_", "i4"), ("vx", "f4"), ("vy", "f4"), ("center_x", "f4"), ("center_y", "f4"), ("id", "u4")],
}

DECODE_DTYPES = {
    0: {"names": ["x", "y", "p", "t"], "formats": ["<u2", "<u2", "<i2", "<i8"], "offsets": [0, 2, 4, 8], "itemsize": 16},
    12: {"names": ["x", "y", "p", "t"], "formats": ["<u2", "<u2", "<i2", "<i8"], "offsets": [0, 2, 4, 8], "itemsize": 16},
    14: {"names": ["p", "t", "id"], "formats": ["<i2", "<i8", "<i2"], "offsets": [0, 8, 16], "itemsize": 24},
    40: {
        "names": ["x", "y", "p", "t", "vx", "vy", "center_x", "center_y", "id"],
        "formats": ["<u2", "<u2", "<i2", "<i8", "f4", "f4", "f4", "f4", "u4"],
        "offsets": [0, 2, 4, 8, 16, 20, 24, 28, 32],
        "itemsize": 36,
    },
}

EV_STRINGS = {0: "Event2D", 12: "EventCD", 14: "EventExtTrigger", 40: "EventOpticalFlow"}

X_MASK = 2**14 - 1
Y_MASK = 2**28 - 2**14
P_MASK = 2**29 - 2**28


def parse_header(f) -> Tuple[int, int, int, list]:
    f.seek(0, os.SEEK_SET)
    end_of_header = False
    header = []
    num_comment_line = 0
    size = [None, None]

    while not end_of_header:
        bod = f.tell()
        line = f.readline()
        first_item = line.decode("latin-1")[:2] if sys.version_info > (3, 0) else line[:2]

        if first_item != "% ":
            end_of_header = True
        else:
            words = line.split()
            if len(words) > 1:
                if words[1] == b"Date" or words[1] == "Date":
                    header += ["Date", words[2] + b" " + words[3] if isinstance(words[2], bytes) else words[2] + " " + words[3]]
                if words[1] == b"Height" or words[1] == "Height":
                    size[0] = int(words[2])
                    header += ["Height", words[2]]
                if words[1] == b"Width" or words[1] == "Width":
                    size[1] = int(words[2])
                    header += ["Width", words[2]]
            else:
                header += words[1:3]
            num_comment_line += 1

    f.seek(bod, os.SEEK_SET)

    if num_comment_line > 0:
        ev_type = np.frombuffer(f.read(1), dtype=np.uint8)[0]
        ev_size = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
    else:
        ev_type = 0
        ev_size = sum(int(n[-1]) for _, n in EV_TYPES[ev_type])

    bod = f.tell()
    return bod, int(ev_type), ev_size, size


def _dat_transfer(dat: np.ndarray, decoded_dtype, xyp=None) -> np.ndarray:
    variables = []
    xyp_index = -1

    for i, name in enumerate(dat.dtype.names):
        if name == "_":
            xyp_index = i
            continue
        variables.append((name, dat[name]))

    if xyp is not None and xyp_index == -1:
        raise ValueError("Decoded event chunk does not contain packed x/y/p field '_'.")

    new_dat = np.empty(dat.shape[0], dtype=decoded_dtype)
    if xyp is not None:
        new_dat["x"] = xyp[0]
        new_dat["y"] = xyp[1]
        new_dat["p"] = xyp[2]

    for name, arr in variables:
        new_dat[name] = arr

    return new_dat


def _decode_packed_fields(dat: np.ndarray, dtype):
    if ("_", "i4") not in dtype:
        return None
    x = np.bitwise_and(dat["_"], X_MASK)
    y = np.right_shift(np.bitwise_and(dat["_"], Y_MASK), 14)
    p = np.right_shift(np.bitwise_and(dat["_"], P_MASK), 28)
    return x, y, p


def load_events_chunked(filename: str, chunk_size: int = 5_000_000, start_event: int = 0) -> Iterator[np.ndarray]:
    with open(filename, "rb") as f:
        body_offset, ev_type, ev_size, _ = parse_header(f)
        dtype = EV_TYPES[ev_type]

        if start_event > 0:
            f.seek(body_offset + int(start_event) * ev_size, os.SEEK_SET)

        while True:
            dat = np.fromfile(f, dtype=dtype, count=chunk_size)
            if len(dat) == 0:
                break
            yield _dat_transfer(dat, DECODE_DTYPES[ev_type], xyp=_decode_packed_fields(dat, dtype))


def load_events(filename: str, ev_count: int = -1, ev_start: int = 0) -> np.ndarray:
    with open(filename, "rb") as f:
        _, ev_type, ev_size, _ = parse_header(f)
        if ev_start > 0:
            f.seek(ev_start * ev_size, os.SEEK_CUR)

        dtype = EV_TYPES[ev_type]
        dat = np.fromfile(f, dtype=dtype, count=ev_count)
        return _dat_transfer(dat, DECODE_DTYPES[ev_type], xyp=_decode_packed_fields(dat, dtype))


def count_events(filename: str) -> int:
    with open(filename, "rb") as f:
        bod, _, ev_size, _ = parse_header(f)
        f.seek(0, os.SEEK_END)
        eod = f.tell()
        if (eod - bod) % ev_size != 0:
            raise ValueError("Unexpected .dat format: file payload is not divisible by event size.")
        return (eod - bod) // ev_size


def read_event_time_at_index(filename: str, event_index: int) -> int | None:
    with open(filename, "rb") as f:
        body_offset, ev_type, ev_size, _ = parse_header(f)
        f.seek(body_offset + int(event_index) * ev_size, os.SEEK_SET)
        event = np.fromfile(f, dtype=EV_TYPES[ev_type], count=1)

        if len(event) == 0:
            return None

        return int(event["t"][0])


def find_event_index_at_or_after_time(filename: str, target_time_us: int) -> int:
    """
    Zwraca indeks pierwszego eventu z timestampem >= target_time_us.
    Używa binary search, więc nie czyta całego pliku od początku.
    """
    n_events = count_events(filename)
    lo = 0
    hi = n_events

    while lo < hi:
        mid = (lo + hi) // 2
        t_mid = read_event_time_at_index(filename, mid)

        if t_mid is None:
            hi = mid
        elif t_mid < target_time_us:
            lo = mid + 1
        else:
            hi = mid

    return int(lo)


def get_recording_duration(dat_path: str) -> Dict[str, float]:
    with open(dat_path, "rb") as f:
        start_pos, ev_type, ev_size, _ = parse_header(f)
        f.seek(0, os.SEEK_END)
        end_pos = f.tell()
        event_count = (end_pos - start_pos) // ev_size
        if event_count <= 0:
            return {"event_count": 0, "start_time_us": 0, "end_time_us": 0, "duration_us": 0, "duration_s": 0.0, "duration_min": 0.0}

        f.seek(start_pos)
        first_event = np.fromfile(f, dtype=EV_TYPES[ev_type], count=1)
        t0 = int(first_event["t"][0])

        f.seek(start_pos + (event_count - 1) * ev_size)
        last_event = np.fromfile(f, dtype=EV_TYPES[ev_type], count=1)
        t1 = int(last_event["t"][0])

    duration_us = int(t1 - t0)
    return {
        "event_count": int(event_count),
        "start_time_us": t0,
        "end_time_us": t1,
        "duration_us": duration_us,
        "duration_s": duration_us / 1e6,
        "duration_min": duration_us / 1e6 / 60,
    }
