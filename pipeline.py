import numpy as np
import pandas as pd

from components import extract_components
from config import PipelineConfig
from bundles import iter_windows_from_bundles
from filters import density_filter_events
from dat import parse_header
from masks import build_closed_mask
from motion_compensation import estimate_rpm_series_for_component
from polarity import select_events_by_polarity
from tracking import initialize_fixed_tracks, assign_components_to_fixed_tracks, update_fixed_tracks


def read_image_shape(dat_path):
    with open(dat_path, "rb") as f:
        _, _, _, size = parse_header(f)

    if size[0] is None or size[1] is None:
        raise ValueError("Could not read sensor size from .dat header")

    return int(size[0]), int(size[1])


def process_window(window_dict, image_shape, config: PipelineConfig):
    xs, ys, ps, _ = select_events_by_polarity(window_dict["events"], config.polarity_mode)

    keep_mask = density_filter_events(
        xs,
        ys,
        image_shape=image_shape,
        bin_size=config.bin_size,
        min_events_in_bin=config.min_events_in_bin,
        min_component_area=config.density_min_component_area,
        remove_border_components=config.remove_border_components,
    )

    xs_f = xs[keep_mask]
    ys_f = ys[keep_mask]

    closed_mask = build_closed_mask(
        xs_f,
        ys_f,
        image_shape=image_shape,
        dilate_kernel_size=config.mask_dilate_kernel_size,
        dilate_iterations=config.mask_dilate_iterations,
        close_kernel_size=config.mask_close_kernel_size,
        close_iterations=config.mask_close_iterations,
        fill_holes=config.mask_fill_holes,
        min_component_area_px=config.mask_min_component_area_px,
        remove_border_components=config.remove_border_components,
    )

    components = extract_components(
        closed_mask,
        bbox_pad=4,
        max_components=config.max_detection_components,
    )

    return {
        "window_idx": window_dict["window_idx"],
        "bundle_idx_start": window_dict["bundle_idx_start"],
        "bundle_idx_end": window_dict["bundle_idx_end"],
        "t0": window_dict["t0"],
        "t1": window_dict["t1"],
        "components": components,
        "closed_mask": closed_mask,
        "n_events_raw": int(len(xs)),
        "n_events_filtered": int(len(xs_f)),
    }


def _empty_output(config: PipelineConfig):
    out = pd.DataFrame(columns=[f"C{i}" for i in range(config.n_propellers)])
    out.attrs["bundle_us"] = config.bundle_us
    out.attrs["bundles_per_window"] = config.bundles_per_window
    out.attrs["window_us"] = config.bundle_us * config.bundles_per_window
    out.attrs["polarity_mode"] = config.polarity_mode
    out.attrs["n_propellers"] = config.n_propellers
    return out


def _filter_params_for_preview(config: PipelineConfig):
    return {
        "bin_size": config.bin_size,
        "min_events_in_bin": config.min_events_in_bin,
        "min_component_area": config.density_min_component_area,
        "remove_border_components": config.remove_border_components,
    }


def _make_rpm_by_track(row, n_propellers):
    return {tid: row.get(tid, np.nan) for tid in range(n_propellers)}


def _show_preview_if_needed(
    config,
    window_counter,
    window_dict,
    result,
    image_shape,
    show_preview,
    matched_components=None,
    rpm_by_track=None,
):
    if not config.preview:
        return True

    if window_counter % config.preview_every_n_windows != 0:
        return True

    return show_preview(
        window_dict,
        result,
        image_shape,
        config.polarity_mode,
        wait_ms=config.preview_wait_ms,
        filter_params=_filter_params_for_preview(config),
        matched_components=matched_components or {},
        rpm_by_track=rpm_by_track or {},
        reference_time_fractions=config.reference_time_fractions,
    )


def run_rpm_pipeline(dat_path, config: PipelineConfig | None = None):
    config = config or PipelineConfig()
    image_shape = read_image_shape(dat_path)

    rows = []
    tracks = None

    if config.preview:
        from visualization import show_preview, close_preview_windows
    else:
        show_preview = None
        close_preview_windows = None

    try:
        for window_counter, window_dict in enumerate(
            iter_windows_from_bundles(
                dat_path,
                chunk_size=config.chunk_size,
                bundle_us=config.bundle_us,
                bundles_per_window=config.bundles_per_window,
                verbose_chunks=config.verbose_chunks,
                start_time_ms=config.start_time_ms,
            )
        ):
            result = process_window(window_dict=window_dict, image_shape=image_shape, config=config)

            row = {"window_idx": result["window_idx"]}
            for tid in range(config.n_propellers):
                row[tid] = np.nan

            matched_components = {}

            if tracks is None:
                tracks = initialize_fixed_tracks(
                    result["components"],
                    n_propellers=config.n_propellers,
                    bootstrap_sort=config.bootstrap_sort,
                )

                if tracks is None:
                    rows.append(row)
                    if config.preview:
                        should_continue = _show_preview_if_needed(
                            config,
                            window_counter,
                            window_dict,
                            result,
                            image_shape,
                            show_preview,
                            matched_components={},
                            rpm_by_track=_make_rpm_by_track(row, config.n_propellers),
                        )
                        if not should_continue:
                            break

                    if config.max_windows is not None and len(rows) >= config.max_windows:
                        break

                    continue

                matched_components = {
                    tid: tracks[tid]["last_component"]
                    for tid in range(config.n_propellers)
                }

            else:
                assignment = assign_components_to_fixed_tracks(
                    components=result["components"],
                    tracks=tracks,
                    max_dx_per_window=config.max_centroid_dx_per_window,
                    max_dy_per_window=config.max_centroid_dy_per_window,
                    max_distance_per_window=config.max_centroid_distance_per_window,
                )

                matched_components, tracks = update_fixed_tracks(
                    result["components"],
                    tracks,
                    assignment,
                )

            for tid in range(config.n_propellers):
                comp = matched_components.get(tid, None)
                if comp is None:
                    continue

                rpm_info = estimate_rpm_series_for_component(
                    window_dict=window_dict,
                    component=comp,
                    track_state=tracks[tid],
                    polarity_mode=config.polarity_mode,
                    rpm_abs_max=config.rpm_abs_max,
                    rpm_step_coarse=config.rpm_step_coarse,
                    rpm_step_fine=config.rpm_step_fine,
                    rpm_refine_span=config.rpm_refine_span,
                    rpm_local_delta=config.rpm_local_delta,
                    full_search_period_us=config.full_search_period_us,
                    min_events_for_rpm=config.min_events_for_rpm,
                    score_mode=config.score_mode,
                    score_lambda=config.mc_score_lambda,
                    score_eps=config.mc_score_eps,
                    max_events_mc=config.max_events_mc,
                    downsample_time_bins=config.downsample_time_bins,
                    reference_time_fractions=config.reference_time_fractions,
                    center_search_radius_px=config.center_search_radius_px,
                    center_search_step_px=config.center_search_step_px,
                )

                rpm = rpm_info["rpm_median"]
                row[tid] = rpm

                comp_for_preview = dict(comp)
                first_est = rpm_info.get("first_bundle_estimate")
                center_est = first_est if first_est is not None else rpm_info

                if center_est is not None and not np.isnan(center_est.get("center_x", np.nan)):
                    comp_for_preview["optimized_center_x"] = float(center_est["center_x"])
                    comp_for_preview["optimized_center_y"] = float(center_est["center_y"])
                    comp_for_preview["optimized_center_dx"] = float(center_est["center_dx"])
                    comp_for_preview["optimized_center_dy"] = float(center_est["center_dy"])
                    comp_for_preview["preview_rpm"] = float(center_est.get("rpm", rpm))
                    comp_for_preview["rpm_median"] = float(rpm) if not np.isnan(rpm) else np.nan

                matched_components[tid] = comp_for_preview

                if not np.isnan(rpm):
                    tracks[tid]["prev_rpm"] = rpm
                    if not np.isnan(rpm_info.get("center_x", np.nan)):
                        tracks[tid]["optimized_center_x"] = float(rpm_info["center_x"])
                        tracks[tid]["optimized_center_y"] = float(rpm_info["center_y"])
                        tracks[tid]["optimized_center_dx"] = float(rpm_info["center_dx"])
                        tracks[tid]["optimized_center_dy"] = float(rpm_info["center_dy"])

                    if tracks[tid]["locked_sign"] is None and rpm != 0:
                        tracks[tid]["locked_sign"] = +1 if rpm > 0 else -1

                    if rpm_info["search_mode"] == "global":
                        tracks[tid]["last_full_search_t0_us"] = result["t0"]

            rows.append(row)

            if config.preview:
                should_continue = _show_preview_if_needed(
                    config,
                    window_counter,
                    window_dict,
                    result,
                    image_shape,
                    show_preview,
                    matched_components=matched_components,
                    rpm_by_track=_make_rpm_by_track(row, config.n_propellers),
                )
                if not should_continue:
                    break

            if config.max_windows is not None and len(rows) >= config.max_windows:
                break

    finally:
        if config.preview and close_preview_windows is not None:
            close_preview_windows()

    if len(rows) == 0:
        return _empty_output(config)

    rpm_df = pd.DataFrame(rows).set_index("window_idx").sort_index()

    for tid in range(config.n_propellers):
        if tid not in rpm_df.columns:
            rpm_df[tid] = np.nan

    rpm_df = rpm_df[[tid for tid in range(config.n_propellers)]]

    rename_map = {}
    for tid in range(config.n_propellers):
        if tracks is not None and tid in tracks and tracks[tid].get("initialized", False):
            rename_map[tid] = f"C{tid}_({tracks[tid]['label_centroid_x']:.1f},{tracks[tid]['label_centroid_y']:.1f})"
        else:
            rename_map[tid] = f"C{tid}"

    rpm_df = rpm_df.rename(columns=rename_map)

    track_meta_rows = []
    for tid in range(config.n_propellers):
        if tracks is not None and tid in tracks:
            st = tracks[tid]
            track_meta_rows.append({
                "track_id": tid,
                "column_name": rename_map[tid],
                "label_centroid_x": st["label_centroid_x"],
                "label_centroid_y": st["label_centroid_y"],
                "last_centroid_x": st["centroid_x"],
                "last_centroid_y": st["centroid_y"],
                "prev_rpm": st["prev_rpm"],
                "locked_sign": st["locked_sign"],
                "last_full_search_t0_us": st["last_full_search_t0_us"],
                "missed_windows": st["missed_windows"],
                "initialized": st["initialized"],
            })
        else:
            track_meta_rows.append({
                "track_id": tid,
                "column_name": rename_map[tid],
                "label_centroid_x": np.nan,
                "label_centroid_y": np.nan,
                "last_centroid_x": np.nan,
                "last_centroid_y": np.nan,
                "prev_rpm": np.nan,
                "locked_sign": None,
                "last_full_search_t0_us": None,
                "missed_windows": np.nan,
                "initialized": False,
            })

    rpm_df.attrs["track_meta"] = pd.DataFrame(track_meta_rows).sort_values("track_id").reset_index(drop=True)
    rpm_df.attrs["bundle_us"] = config.bundle_us
    rpm_df.attrs["bundles_per_window"] = config.bundles_per_window
    rpm_df.attrs["window_us"] = config.bundle_us * config.bundles_per_window
    rpm_df.attrs["polarity_mode"] = config.polarity_mode
    rpm_df.attrs["n_propellers"] = config.n_propellers

    return rpm_df
