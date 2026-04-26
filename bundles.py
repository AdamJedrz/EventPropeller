from typing import Iterator

import numpy as np

from dat import (
    load_events_chunked,
    get_recording_duration,
    find_event_index_at_or_after_time,
)


def iter_bundles_from_chunks(
    dat_path: str,
    chunk_size: int = 5_000_000,
    bundle_us: int = 1000,
    verbose_chunks: bool = False,
    start_time_ms: float = 0.0,
) -> Iterator[dict]:
    duration = get_recording_duration(dat_path)
    recording_t0_us = int(duration["start_time_us"])

    start_offset_us = max(0, int(round(float(start_time_ms) * 1000.0)))
    start_abs_time_us = recording_t0_us + start_offset_us

    if start_offset_us > 0:
        start_event = find_event_index_at_or_after_time(dat_path, start_abs_time_us)
    else:
        start_event = 0

    carry = None

    bundle_idx = start_offset_us // int(bundle_us)
    current_bundle_start = start_abs_time_us

    if verbose_chunks and start_offset_us > 0:
        print(
            f"[start] start_time_ms={start_time_ms} | "
            f"start_event={start_event} | "
            f"start_abs_time_us={start_abs_time_us}"
        )

    for chunk_idx, chunk in enumerate(
        load_events_chunked(dat_path, chunk_size=chunk_size, start_event=start_event),
        start=1,
    ):
        if verbose_chunks:
            print(f"[chunk {chunk_idx}] start | liczba eventów w chunku: {len(chunk)}")

        if len(chunk) == 0:
            continue

        data = np.concatenate([carry, chunk]) if carry is not None and len(carry) > 0 else chunk
        if len(data) == 0:
            continue

        while True:
            if len(data) == 0:
                carry = data
                break

            if int(data["t"][0]) >= current_bundle_start + bundle_us:
                current_bundle_start = int(data["t"][0])
                bundle_idx = max(bundle_idx, (current_bundle_start - recording_t0_us) // int(bundle_us))

            current_bundle_end = current_bundle_start + bundle_us
            end_idx = np.searchsorted(data["t"], current_bundle_end, side="left")

            if end_idx == len(data) and int(data["t"][-1]) < current_bundle_end:
                carry = data
                break

            yield {
                "bundle_idx": int(bundle_idx),
                "t0": int(current_bundle_start),
                "t1": int(current_bundle_end),
                "events": data[:end_idx],
            }

            bundle_idx += 1
            data = data[end_idx:]
            current_bundle_start = current_bundle_end


def iter_windows_from_bundles(
    dat_path: str,
    chunk_size: int = 5_000_000,
    bundle_us: int = 1000,
    bundles_per_window: int = 1,
    verbose_chunks: bool = False,
    start_time_ms: float = 0.0,
) -> Iterator[dict]:
    buffer = []

    for bundle in iter_bundles_from_chunks(
        dat_path,
        chunk_size=chunk_size,
        bundle_us=bundle_us,
        verbose_chunks=verbose_chunks,
        start_time_ms=start_time_ms,
    ):
        buffer.append(bundle)
        if len(buffer) < bundles_per_window:
            continue

        yield {
            "window_idx": buffer[0]["bundle_idx"],
            "bundle_idx_start": buffer[0]["bundle_idx"],
            "bundle_idx_end": buffer[-1]["bundle_idx"],
            "t0": buffer[0]["t0"],
            "t1": buffer[-1]["t1"],
            "events": np.concatenate([b["events"] for b in buffer]),
            "bundles": list(buffer),
        }
        buffer = []
