import numpy as np
import pandas as pd

from components import extract_propeller_clusters_from_events
from config import PipelineConfig
from bundles import iter_windows_from_bundles
from filters import (
    density_filter_events,
    apply_bin_morphology,
    keep_mask_from_active_bins,
)
from dat import parse_header
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

    density_keep_mask, active_bins = density_filter_events(
        xs,
        ys,
        image_shape=image_shape,
        bin_size=config.bin_size,
        min_events_in_bin=config.min_events_in_bin,
        min_component_area=config.density_min_component_area,
        remove_border_components=config.remove_border_components,
        return_active_bins=True,
    )

    processed_bins = apply_bin_morphology(
        active_bins,
        mode=config.density_bin_morph_mode,
        kernel_size=config.density_bin_morph_kernel_size,
        iterations=config.density_bin_morph_iterations,
    )

    precluster_keep_mask = keep_mask_from_active_bins(
        xs,
        ys,
        image_shape=image_shape,
        bin_size=config.bin_size,
        active_bins=processed_bins,
    )

    xs_precluster = xs[precluster_keep_mask]
    ys_precluster = ys[precluster_keep_mask]

    components, cluster_debug = extract_propeller_clusters_from_events(
        xs_precluster,
        ys_precluster,
        image_shape=image_shape,
        n_components=config.n_propellers,
        bbox_pad=config.cluster_bbox_pad,
        min_events_per_cluster=config.cluster_min_events_per_cluster,
        sample_max_events=config.cluster_sample_max_events,
        attempts=config.cluster_attempts,
    )

    return {
        "window_idx": window_dict["window_idx"],
        "bundle_idx_start": window_dict["bundle_idx_start"],
        "bundle_idx_end": window_dict["bundle_idx_end"],
        "t0": window_dict["t0"],
        "t1": window_dict["t1"],
        "components": components,
        "density_keep_mask": density_keep_mask,
        "precluster_keep_mask": precluster_keep_mask,
        "active_bins": active_bins,
        "processed_bins": processed_bins,
        "cluster_debug": cluster_debug,
        "n_events_raw": int(len(xs)),
        "n_events_filtered": int(np.count_nonzero(density_keep_mask)),
        "n_events_precluster": int(np.count_nonzero(precluster_keep_mask)),
        "n_events_clustered": int(len(cluster_debug.get("assigned_xs", []))),
        "n_events_unassigned": int(len(cluster_debug.get("unassigned_xs", []))),
    }

def _ordered_internal_cols(config: PipelineConfig):
    cols = []
    for tid in range(config.n_propellers):
        cols.extend([tid, f"q_{tid}", f"axis_{tid}"])
    return cols


def _empty_output(config: PipelineConfig):
    out = pd.DataFrame(columns=_ordered_internal_cols(config))
    out.attrs["bundle_us"] = config.bundle_us
    out.attrs["bundles_per_window"] = config.bundles_per_window
    out.attrs["window_us"] = config.bundle_us * config.bundles_per_window
    out.attrs["polarity_mode"] = config.polarity_mode
    out.attrs["n_propellers"] = config.n_propellers
    return out


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
        matched_components=matched_components or {},
        rpm_by_track=rpm_by_track or {},
        reference_time_fractions=config.reference_time_fractions,
    )


def run_rpm_pipeline(dat_path, config: PipelineConfig | None = None):
    config = config or PipelineConfig()
    image_shape = read_image_shape(dat_path)

    rows = []
    tracks = None
    interrupted = False

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
                row[f"q_{tid}"] = np.nan
                row[f"axis_{tid}"] = np.nan

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
                    parallel_mc=config.parallel_mc,
                    parallel_candidate_chunk_size=config.parallel_candidate_chunk_size,
                    reference_time_fractions=config.reference_time_fractions,
                    center_search_radius_px=config.center_search_radius_px,
                    center_search_step_px=config.center_search_step_px,
                    q_search_enabled=config.q_search_enabled,
                    q_fixed=config.q_fixed,
                    q_min=config.q_min,
                    q_max=config.q_max,
                    q_step=config.q_step,
                    q_axis_angle_deg=config.q_axis_angle_deg,
                    q_axis_search_enabled=config.q_axis_search_enabled,
                    q_axis_angle_min_deg=config.q_axis_angle_min_deg,
                    q_axis_angle_max_deg=config.q_axis_angle_max_deg,
                    q_axis_angle_step_deg=config.q_axis_angle_step_deg,
                )

                rpm = rpm_info["rpm_median"]
                row[tid] = rpm
                row[f"q_{tid}"] = float(rpm_info.get("q", np.nan))
                row[f"axis_{tid}"] = float(rpm_info.get("q_axis_angle_deg", np.nan))

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
                    comp_for_preview["optimized_q"] = float(center_est.get("q", config.q_fixed))
                    comp_for_preview["q_axis_angle_deg"] = float(center_est.get("q_axis_angle_deg", config.q_axis_angle_deg))

                matched_components[tid] = comp_for_preview

                if not np.isnan(rpm):
                    tracks[tid]["prev_rpm"] = rpm
                    if not np.isnan(rpm_info.get("center_x", np.nan)):
                        tracks[tid]["optimized_center_x"] = float(rpm_info["center_x"])
                        tracks[tid]["optimized_center_y"] = float(rpm_info["center_y"])
                        tracks[tid]["optimized_center_dx"] = float(rpm_info["center_dx"])
                        tracks[tid]["optimized_center_dy"] = float(rpm_info["center_dy"])
                        tracks[tid]["optimized_q"] = float(rpm_info.get("q", np.nan))
                        tracks[tid]["optimized_q_axis_angle_deg"] = float(rpm_info.get("q_axis_angle_deg", np.nan))

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

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Przerwano Ctrl+C — zwracam dotychczas policzone wyniki z RAM.")

    finally:
        if config.preview and close_preview_windows is not None:
            close_preview_windows()

    if len(rows) == 0:
        out = _empty_output(config)
        out.attrs["interrupted"] = interrupted
        return out

    rpm_df = pd.DataFrame(rows).set_index("window_idx").sort_index()

    for col in _ordered_internal_cols(config):
        if col not in rpm_df.columns:
            rpm_df[col] = np.nan

    rpm_df = rpm_df[_ordered_internal_cols(config)]

    rename_map = {}
    for tid in range(config.n_propellers):
        if tracks is not None and tid in tracks and tracks[tid].get("initialized", False):
            rename_map[tid] = f"C{tid}_({tracks[tid]['label_centroid_x']:.1f},{tracks[tid]['label_centroid_y']:.1f})"
        else:
            rename_map[tid] = f"C{tid}"
        rename_map[f"q_{tid}"] = f"Q{tid}"
        rename_map[f"axis_{tid}"] = f"A{tid}"

    rpm_df = rpm_df.rename(columns=rename_map)

    track_meta_rows = []
    for tid in range(config.n_propellers):
        if tracks is not None and tid in tracks:
            st = tracks[tid]
            track_meta_rows.append({
                "track_id": tid,
                "column_name": rename_map[tid],
                "q_column_name": rename_map[f"q_{tid}"],
                "axis_column_name": rename_map[f"axis_{tid}"],
                "label_centroid_x": st["label_centroid_x"],
                "label_centroid_y": st["label_centroid_y"],
                "last_centroid_x": st["centroid_x"],
                "last_centroid_y": st["centroid_y"],
                "prev_rpm": st["prev_rpm"],
                "locked_sign": st["locked_sign"],
                "last_full_search_t0_us": st["last_full_search_t0_us"],
                "missed_windows": st["missed_windows"],
                "initialized": st["initialized"],
                "optimized_q": st.get("optimized_q", np.nan),
                "optimized_q_axis_angle_deg": st.get("optimized_q_axis_angle_deg", np.nan),
            })
        else:
            track_meta_rows.append({
                "track_id": tid,
                "column_name": rename_map[tid],
                "q_column_name": rename_map[f"q_{tid}"],
                "axis_column_name": rename_map[f"axis_{tid}"],
                "label_centroid_x": np.nan,
                "label_centroid_y": np.nan,
                "last_centroid_x": np.nan,
                "last_centroid_y": np.nan,
                "prev_rpm": np.nan,
                "locked_sign": None,
                "last_full_search_t0_us": None,
                "missed_windows": np.nan,
                "initialized": False,
                "optimized_q": np.nan,
                "optimized_q_axis_angle_deg": np.nan,
            })

    rpm_df.attrs["track_meta"] = pd.DataFrame(track_meta_rows).sort_values("track_id").reset_index(drop=True)
    rpm_df.attrs["bundle_us"] = config.bundle_us
    rpm_df.attrs["bundles_per_window"] = config.bundles_per_window
    rpm_df.attrs["window_us"] = config.bundle_us * config.bundles_per_window
    rpm_df.attrs["polarity_mode"] = config.polarity_mode
    rpm_df.attrs["n_propellers"] = config.n_propellers
    rpm_df.attrs["interrupted"] = interrupted

    return rpm_df
