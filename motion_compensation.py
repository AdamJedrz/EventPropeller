import numpy as np

from polarity import select_events_by_polarity


def crop_events_to_component_bbox(xs, ys, ps, ts, component):
    x0, y0 = component["bbox_x"], component["bbox_y"]
    x1 = x0 + component["bbox_w"]
    y1 = y0 + component["bbox_h"]

    mask = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    return xs[mask], ys[mask], ps[mask], ts[mask]


def downsample_events_time_stratified(xs, ys, ps, ts, max_events=3000, n_time_bins=20):
    n = len(xs)
    if max_events is None or n <= max_events:
        return xs, ys, ps, ts

    n_time_bins = 20 if n_time_bins is None or n_time_bins < 1 else int(n_time_bins)
    t_min = int(ts.min())
    t_max = int(ts.max())

    if t_max <= t_min:
        idx = np.linspace(0, n - 1, max_events).astype(np.int32)
        return xs[idx], ys[idx], ps[idx], ts[idx]

    edges = np.linspace(t_min, t_max + 1, n_time_bins + 1)
    per_bin = max(1, int(np.ceil(max_events / n_time_bins)))
    selected_parts = []

    for i in range(n_time_bins):
        if i == n_time_bins - 1:
            mask = (ts >= edges[i]) & (ts <= t_max)
        else:
            mask = (ts >= edges[i]) & (ts < edges[i + 1])

        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue

        if len(idx) > per_bin:
            idx = idx[np.linspace(0, len(idx) - 1, per_bin).astype(np.int32)]

        selected_parts.append(idx)

    if len(selected_parts) == 0:
        idx = np.linspace(0, n - 1, max_events).astype(np.int32)
    else:
        idx = np.unique(np.concatenate(selected_parts))
        if len(idx) > max_events:
            idx = idx[np.linspace(0, len(idx) - 1, max_events).astype(np.int32)]

    return xs[idx], ys[idx], ps[idx], ts[idx]


def warp_events_about_centroid(xs, ys, ts, cx, cy, omega_rad_s, t_ref_us):
    dt = (float(t_ref_us) - ts.astype(np.float64)) * 1e-6
    ang = omega_rad_s * dt

    dx = xs.astype(np.float64) - float(cx)
    dy = ys.astype(np.float64) - float(cy)

    c = np.cos(ang)
    s = np.sin(ang)

    xw = float(cx) + c * dx - s * dy
    yw = float(cy) + s * dx + c * dy

    return xw, yw


def build_local_histogram(xw, yw, component):
    width = component["bbox_w"]
    height = component["bbox_h"]
    x0 = component["bbox_x"]
    y0 = component["bbox_y"]

    xi = np.rint(xw).astype(np.int32) - x0
    yi = np.rint(yw).astype(np.int32) - y0

    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

    hist = np.zeros((height, width), dtype=np.int32)
    if np.any(valid):
        np.add.at(hist, (yi[valid], xi[valid]), 1)

    return hist


def mean_square_objective(hist):
    if hist.sum() <= 0:
        return -np.inf
    h = hist.astype(np.float64)
    return float(np.mean(h * h))


def eventpro_objective(hist, score_lambda=1.0, score_eps=1e-6):
    h = hist.astype(np.float64)
    if h.sum() <= 0:
        return -np.inf

    exp_h = np.exp(np.clip(h, 0, 20))
    r_acc = exp_h.sum()
    r_spa = (1.0 / (exp_h - 1.0 + score_eps)).sum()

    return float(r_acc + score_lambda * r_spa)


def score_histogram(hist, score_mode="mean_square", score_lambda=1.0, score_eps=1e-6):
    if score_mode == "mean_square":
        return mean_square_objective(hist)
    if score_mode == "eventpro":
        return eventpro_objective(hist, score_lambda=score_lambda, score_eps=score_eps)
    raise ValueError("score_mode musi być 'mean_square' albo 'eventpro'")


def make_reference_times_us(t0_us, t1_us, reference_time_fractions):
    t0_us = float(t0_us)
    t1_us = float(t1_us)
    duration_us = t1_us - t0_us
    return [t0_us + float(frac) * duration_us for frac in reference_time_fractions]


def make_center_offsets(radius_px=0.0, step_px=1.0):
    radius_px = float(radius_px)
    step_px = float(step_px)

    if radius_px <= 0 or step_px <= 0:
        return [(0.0, 0.0)]

    values = np.arange(-radius_px, radius_px + 0.5 * step_px, step_px, dtype=np.float64)
    return [(float(dx), float(dy)) for dy in values for dx in values]


def score_rpm_candidate(
    xs,
    ys,
    ts,
    component,
    rpm,
    reference_times_us,
    center_offsets,
    score_mode="mean_square",
    score_lambda=1.0,
    score_eps=1e-6,
):
    omega = float(rpm) * 2.0 * np.pi / 60.0

    base_cx = float(component["centroid_x"])
    base_cy = float(component["centroid_y"])

    best = {
        "score": -np.inf,
        "center_x": base_cx,
        "center_y": base_cy,
        "center_dx": 0.0,
        "center_dy": 0.0,
    }

    for dx, dy in center_offsets:
        cx = base_cx + dx
        cy = base_cy + dy

        scores = []
        for t_ref_us in reference_times_us:
            xw, yw = warp_events_about_centroid(xs, ys, ts, cx, cy, omega, t_ref_us)
            hist = build_local_histogram(xw, yw, component)
            scores.append(score_histogram(hist, score_mode=score_mode, score_lambda=score_lambda, score_eps=score_eps))

        total_score = float(np.mean(scores))
        if total_score > best["score"]:
            best = {
                "score": total_score,
                "center_x": float(cx),
                "center_y": float(cy),
                "center_dx": float(dx),
                "center_dy": float(dy),
            }

    return best


def _nan_estimate():
    return {
        "rpm": np.nan,
        "score": -np.inf,
        "center_x": np.nan,
        "center_y": np.nan,
        "center_dx": np.nan,
        "center_dy": np.nan,
    }


def estimate_rpm_for_component_on_arrays(
    xs,
    ys,
    ps,
    ts,
    t0_us,
    t1_us,
    component,
    rpm_candidates,
    min_events_for_rpm=50,
    score_mode="mean_square",
    score_lambda=1.0,
    score_eps=1e-6,
    max_events_mc=3000,
    downsample_time_bins=20,
    reference_time_fractions=(0.5,),
    center_search_radius_px=0.0,
    center_search_step_px=1.0,
    refine=False,
    rpm_step_fine=20,
    rpm_refine_span=200,
    sign_lock=None,
):
    xs, ys, ps, ts = crop_events_to_component_bbox(xs, ys, ps, ts, component)
    if len(xs) < min_events_for_rpm:
        return _nan_estimate()

    xs, ys, ps, ts = downsample_events_time_stratified(
        xs,
        ys,
        ps,
        ts,
        max_events=max_events_mc,
        n_time_bins=downsample_time_bins,
    )
    if len(xs) < min_events_for_rpm:
        return _nan_estimate()

    rpm_candidates = np.asarray(rpm_candidates, dtype=np.float64)
    if len(rpm_candidates) == 0:
        return _nan_estimate()

    reference_times_us = make_reference_times_us(t0_us, t1_us, reference_time_fractions)
    center_offsets = make_center_offsets(center_search_radius_px, center_search_step_px)

    def eval_rpm(rpm):
        best_for_rpm = score_rpm_candidate(
            xs=xs,
            ys=ys,
            ts=ts,
            component=component,
            rpm=rpm,
            reference_times_us=reference_times_us,
            center_offsets=center_offsets,
            score_mode=score_mode,
            score_lambda=score_lambda,
            score_eps=score_eps,
        )
        best_for_rpm["rpm"] = float(rpm)
        return best_for_rpm

    coarse_results = [eval_rpm(rpm) for rpm in rpm_candidates]
    coarse_scores = np.array([r["score"] for r in coarse_results], dtype=np.float64)
    if np.all(~np.isfinite(coarse_scores)):
        return _nan_estimate()

    best_result = dict(coarse_results[int(np.nanargmax(coarse_scores))])
    best_coarse_rpm = float(best_result["rpm"])

    if not refine:
        return best_result

    rpm_min = best_coarse_rpm - rpm_refine_span
    rpm_max = best_coarse_rpm + rpm_refine_span

    if sign_lock is not None:
        if sign_lock > 0:
            rpm_min = max(0.0, rpm_min)
            rpm_max = max(0.0, rpm_max)
        else:
            rpm_min = min(0.0, rpm_min)
            rpm_max = min(0.0, rpm_max)

    if rpm_min > rpm_max:
        rpm_min, rpm_max = rpm_max, rpm_min

    fine_rpms = np.arange(rpm_min, rpm_max + rpm_step_fine, rpm_step_fine, dtype=np.float64)
    if len(fine_rpms) == 0:
        return best_result

    fine_results = [eval_rpm(rpm) for rpm in fine_rpms]
    fine_scores = np.array([r["score"] for r in fine_results], dtype=np.float64)
    if np.all(~np.isfinite(fine_scores)):
        return best_result

    return dict(fine_results[int(np.nanargmax(fine_scores))])


def estimate_rpm_series_for_component(
    window_dict,
    component,
    track_state,
    polarity_mode="positive",
    rpm_abs_max=8000,
    rpm_step_coarse=200,
    rpm_step_fine=20,
    rpm_refine_span=200,
    rpm_local_delta=150,
    full_search_period_us=500_000,
    min_events_for_rpm=50,
    score_mode="mean_square",
    score_lambda=1.0,
    score_eps=1e-6,
    max_events_mc=3000,
    downsample_time_bins=20,
    reference_time_fractions=(0.5,),
    center_search_radius_px=0.0,
    center_search_step_px=1.0,
):
    prev_rpm = track_state.get("prev_rpm", np.nan)
    locked_sign = track_state.get("locked_sign", None)
    last_full_search_t0_us = track_state.get("last_full_search_t0_us", None)

    do_full_search = (
        locked_sign is None
        or prev_rpm is None
        or np.isnan(prev_rpm)
        or last_full_search_t0_us is None
        or (window_dict["t0"] - last_full_search_t0_us >= full_search_period_us)
    )

    if do_full_search:
        search_mode = "global"
        refine = True

        if locked_sign is None:
            rpm_candidates = np.arange(-rpm_abs_max, rpm_abs_max + rpm_step_coarse, rpm_step_coarse, dtype=np.float64)
            sign_lock = None
        elif locked_sign > 0:
            rpm_candidates = np.arange(0, rpm_abs_max + rpm_step_coarse, rpm_step_coarse, dtype=np.float64)
            sign_lock = +1
        else:
            rpm_candidates = np.arange(-rpm_abs_max, 0 + rpm_step_coarse, rpm_step_coarse, dtype=np.float64)
            sign_lock = -1

    else:
        search_mode = "local"
        refine = False
        sign_lock = +1 if locked_sign > 0 else -1

        rpm_min = float(prev_rpm) - rpm_local_delta
        rpm_max = float(prev_rpm) + rpm_local_delta

        if sign_lock > 0:
            rpm_min = max(0.0, rpm_min)
            rpm_max = max(0.0, rpm_max)
        else:
            rpm_min = min(0.0, rpm_min)
            rpm_max = min(0.0, rpm_max)

        if rpm_min > rpm_max:
            rpm_min, rpm_max = rpm_max, rpm_min

        rpm_candidates = np.arange(rpm_min, rpm_max + rpm_step_fine, rpm_step_fine, dtype=np.float64)
        if len(rpm_candidates) == 0:
            rpm_candidates = np.array([float(prev_rpm)], dtype=np.float64)

    estimates = []
    first_bundle_estimate = None

    for bundle_i, bundle_dict in enumerate(window_dict["bundles"]):
        xs, ys, ps, ts = select_events_by_polarity(bundle_dict["events"], polarity_mode)

        est = estimate_rpm_for_component_on_arrays(
            xs=xs,
            ys=ys,
            ps=ps,
            ts=ts,
            t0_us=bundle_dict["t0"],
            t1_us=bundle_dict["t1"],
            component=component,
            rpm_candidates=rpm_candidates,
            min_events_for_rpm=min_events_for_rpm,
            score_mode=score_mode,
            score_lambda=score_lambda,
            score_eps=score_eps,
            max_events_mc=max_events_mc,
            downsample_time_bins=downsample_time_bins,
            reference_time_fractions=reference_time_fractions,
            center_search_radius_px=center_search_radius_px,
            center_search_step_px=center_search_step_px,
            refine=refine,
            rpm_step_fine=rpm_step_fine,
            rpm_refine_span=rpm_refine_span,
            sign_lock=sign_lock,
        )

        est["bundle_idx"] = bundle_dict["bundle_idx"]
        est["t0"] = bundle_dict["t0"]
        est["t1"] = bundle_dict["t1"]

        if not np.isnan(est["rpm"]):
            estimates.append(est)
            if bundle_i == 0:
                first_bundle_estimate = est

    if len(estimates) == 0:
        return {
            "rpm_median": np.nan,
            "search_mode": search_mode,
            "n_valid": 0,
            "center_x": np.nan,
            "center_y": np.nan,
            "center_dx": np.nan,
            "center_dy": np.nan,
            "first_bundle_estimate": None,
            "bundle_estimates": [],
        }

    rpm_values = np.array([e["rpm"] for e in estimates], dtype=np.float64)
    rpm_median = float(np.median(rpm_values))

    best_idx = int(np.nanargmin(np.abs(rpm_values - rpm_median)))
    representative = estimates[best_idx]

    return {
        "rpm_median": rpm_median,
        "search_mode": search_mode,
        "n_valid": len(estimates),
        "center_x": float(representative["center_x"]),
        "center_y": float(representative["center_y"]),
        "center_dx": float(representative["center_dx"]),
        "center_dy": float(representative["center_dy"]),
        "first_bundle_estimate": first_bundle_estimate,
        "bundle_estimates": estimates,
    }
