import numpy as np


def _nan_estimate():
    return {
        "rpm": np.nan,
        "score": -np.inf,
        "center_x": np.nan,
        "center_y": np.nan,
        "center_dx": np.nan,
        "center_dy": np.nan,
    }


def _select_cuda_device(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. parallel_mc wymaga PyTorch z obsługą CUDA.")
    return torch.device("cuda")


def _make_reference_times_us(t0_us, t1_us, reference_time_fractions):
    t0_us = float(t0_us)
    t1_us = float(t1_us)
    duration_us = t1_us - t0_us
    return [t0_us + float(frac) * duration_us for frac in reference_time_fractions]


def _make_center_offsets(radius_px=0.0, step_px=1.0):
    radius_px = float(radius_px)
    step_px = float(step_px)

    if radius_px <= 0 or step_px <= 0:
        return np.array([[0.0, 0.0]], dtype=np.float32)

    values = np.arange(-radius_px, radius_px + 0.5 * step_px, step_px, dtype=np.float32)
    offsets = [(float(dx), float(dy)) for dy in values for dx in values]
    return np.asarray(offsets, dtype=np.float32)


def _score_candidates_torch(
    xs_np,
    ys_np,
    ts_np,
    component,
    rpm_candidates_np,
    center_offsets_np,
    reference_times_us,
    score_mode,
    score_lambda,
    score_eps,
    candidate_chunk_size=512,
):
    import torch

    device = _select_cuda_device(torch)

    xs = torch.as_tensor(xs_np, dtype=torch.float32, device=device)
    ys = torch.as_tensor(ys_np, dtype=torch.float32, device=device)
    ts = torch.as_tensor(ts_np, dtype=torch.float32, device=device)

    rpm_candidates_np = np.asarray(rpm_candidates_np, dtype=np.float32)
    center_offsets_np = np.asarray(center_offsets_np, dtype=np.float32)

    n_rpm = len(rpm_candidates_np)
    n_center = len(center_offsets_np)
    if n_rpm == 0 or n_center == 0 or xs.numel() == 0:
        return _nan_estimate()

    rpm_all = np.repeat(rpm_candidates_np, n_center).astype(np.float32)
    center_dx_all = np.tile(center_offsets_np[:, 0], n_rpm).astype(np.float32)
    center_dy_all = np.tile(center_offsets_np[:, 1], n_rpm).astype(np.float32)

    width = int(component["bbox_w"])
    height = int(component["bbox_h"])
    x0 = float(component["bbox_x"])
    y0 = float(component["bbox_y"])
    hist_size = int(width * height)

    base_cx = float(component["centroid_x"])
    base_cy = float(component["centroid_y"])

    best_score = -np.inf
    best_idx_global = -1

    total_candidates = len(rpm_all)
    chunk_size = max(1, int(candidate_chunk_size))

    for start in range(0, total_candidates, chunk_size):
        end = min(start + chunk_size, total_candidates)
        m = end - start

        rpm = torch.as_tensor(rpm_all[start:end], dtype=torch.float32, device=device)
        center_dx = torch.as_tensor(center_dx_all[start:end], dtype=torch.float32, device=device)
        center_dy = torch.as_tensor(center_dy_all[start:end], dtype=torch.float32, device=device)

        cx = base_cx + center_dx
        cy = base_cy + center_dy
        omega = rpm * (2.0 * np.pi / 60.0)

        candidate_scores = torch.zeros((m,), dtype=torch.float32, device=device)

        for t_ref_us in reference_times_us:
            dt = (float(t_ref_us) - ts) * 1e-6
            ang = omega[:, None] * dt[None, :]

            c = torch.cos(ang)
            s = torch.sin(ang)

            dx = xs[None, :] - cx[:, None]
            dy = ys[None, :] - cy[:, None]

            xw = cx[:, None] + c * dx - s * dy
            yw = cy[:, None] + s * dx + c * dy

            xi = torch.round(xw).to(torch.int64) - int(round(x0))
            yi = torch.round(yw).to(torch.int64) - int(round(y0))

            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

            if not torch.any(valid):
                candidate_scores += -float("inf")
                continue

            cand_ids = torch.arange(m, device=device, dtype=torch.int64)[:, None].expand_as(xi)
            flat_idx = cand_ids * hist_size + yi * width + xi
            flat_idx = flat_idx[valid]

            hist_flat = torch.zeros((m * hist_size,), dtype=torch.float32, device=device)
            hist_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
            hist = hist_flat.reshape(m, hist_size)

            non_empty = hist.sum(dim=1) > 0
            if score_mode == "mean_square":
                scores = (hist * hist).mean(dim=1)
            elif score_mode == "eventpro":
                exp_h = torch.exp(torch.clamp(hist, 0.0, 20.0))
                r_acc = exp_h.sum(dim=1)
                r_spa = (1.0 / (exp_h - 1.0 + float(score_eps))).sum(dim=1)
                scores = r_acc + float(score_lambda) * r_spa
            else:
                raise ValueError("score_mode musi być 'mean_square' albo 'eventpro'")

            scores = torch.where(non_empty, scores, torch.full_like(scores, -float("inf")))
            candidate_scores += scores

        candidate_scores = candidate_scores / max(1, len(reference_times_us))

        score_value, local_idx = torch.max(candidate_scores, dim=0)
        score_float = float(score_value.detach().cpu().item())

        if score_float > best_score:
            best_score = score_float
            best_idx_global = start + int(local_idx.detach().cpu().item())

    if best_idx_global < 0 or not np.isfinite(best_score):
        return _nan_estimate()

    best_rpm = float(rpm_all[best_idx_global])
    best_dx = float(center_dx_all[best_idx_global])
    best_dy = float(center_dy_all[best_idx_global])

    return {
        "rpm": best_rpm,
        "score": float(best_score),
        "center_x": float(base_cx + best_dx),
        "center_y": float(base_cy + best_dy),
        "center_dx": best_dx,
        "center_dy": best_dy,
    }


def estimate_rpm_for_component_on_arrays_torch(
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
    candidate_chunk_size=512,
    already_prepared=False,
):
    if not already_prepared:
        from motion_compensation import crop_events_to_component_bbox, downsample_events_time_stratified

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

    rpm_candidates = np.asarray(rpm_candidates, dtype=np.float32)
    if len(rpm_candidates) == 0:
        return _nan_estimate()

    reference_times_us = _make_reference_times_us(t0_us, t1_us, reference_time_fractions)
    center_offsets = _make_center_offsets(center_search_radius_px, center_search_step_px)

    best_result = _score_candidates_torch(
        xs_np=xs,
        ys_np=ys,
        ts_np=ts,
        component=component,
        rpm_candidates_np=rpm_candidates,
        center_offsets_np=center_offsets,
        reference_times_us=reference_times_us,
        score_mode=score_mode,
        score_lambda=score_lambda,
        score_eps=score_eps,
        candidate_chunk_size=candidate_chunk_size,
    )

    if not refine or np.isnan(best_result["rpm"]):
        return best_result

    best_coarse_rpm = float(best_result["rpm"])
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

    fine_rpms = np.arange(rpm_min, rpm_max + rpm_step_fine, rpm_step_fine, dtype=np.float32)
    if len(fine_rpms) == 0:
        return best_result

    fine_result = _score_candidates_torch(
        xs_np=xs,
        ys_np=ys,
        ts_np=ts,
        component=component,
        rpm_candidates_np=fine_rpms,
        center_offsets_np=center_offsets,
        reference_times_us=reference_times_us,
        score_mode=score_mode,
        score_lambda=score_lambda,
        score_eps=score_eps,
        candidate_chunk_size=candidate_chunk_size,
    )

    if np.isnan(fine_result["rpm"]):
        return best_result

    return fine_result
