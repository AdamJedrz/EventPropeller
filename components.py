import numpy as np
import cv2


def _clip_bbox_from_points(x_values, y_values, width, height, bbox_pad):
    x_min = int(np.floor(np.min(x_values)))
    x_max = int(np.ceil(np.max(x_values)))
    y_min = int(np.floor(np.min(y_values)))
    y_max = int(np.ceil(np.max(y_values)))

    pad = int(max(0, bbox_pad))
    x0 = max(0, x_min - pad)
    y0 = max(0, y_min - pad)
    x1 = min(width - 1, x_max + pad)
    y1 = min(height - 1, y_max + pad)

    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def _valid_event_points(xs, ys, image_shape):
    height, width = image_shape
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return xs[valid].astype(np.float32), ys[valid].astype(np.float32)


def _empty_cluster_debug():
    return {
        "candidate_xs": np.array([], dtype=np.float32),
        "candidate_ys": np.array([], dtype=np.float32),
        "assigned_xs": np.array([], dtype=np.float32),
        "assigned_ys": np.array([], dtype=np.float32),
        "assigned_labels": np.array([], dtype=np.int32),
        "unassigned_xs": np.array([], dtype=np.float32),
        "unassigned_ys": np.array([], dtype=np.float32),
        "centers": np.empty((0, 2), dtype=np.float32),
    }


def extract_propeller_clusters_from_events(
    xs,
    ys,
    image_shape,
    n_components=4,
    bbox_pad=12,
    min_events_per_cluster=40,
    sample_max_events=12000,
    attempts=5,
    random_seed=0,
):

    height, width = image_shape
    n_components = int(n_components)
    min_events_per_cluster = int(min_events_per_cluster)

    debug_empty = _empty_cluster_debug()

    if n_components < 1:
        return [], debug_empty

    xs_all, ys_all = _valid_event_points(xs, ys, image_shape=image_shape)
    n_all = len(xs_all)
    min_total = n_components * max(1, min_events_per_cluster)

    if n_all < min_total:
        debug_empty["candidate_xs"] = xs_all
        debug_empty["candidate_ys"] = ys_all
        debug_empty["unassigned_xs"] = xs_all
        debug_empty["unassigned_ys"] = ys_all
        return [], debug_empty

    xs_fit_full = xs_all
    ys_fit_full = ys_all

    max_events = int(sample_max_events or 0)
    if max_events > 0 and len(xs_fit_full) > max_events:
        rng = np.random.default_rng(int(random_seed))
        sample_idx = rng.choice(len(xs_fit_full), size=max_events, replace=False)
        xs_fit = xs_fit_full[sample_idx]
        ys_fit = ys_fit_full[sample_idx]
    else:
        xs_fit = xs_fit_full
        ys_fit = ys_fit_full

    points_fit = np.column_stack([xs_fit, ys_fit]).astype(np.float32)
    if len(points_fit) < n_components:
        debug_empty["candidate_xs"] = xs_all
        debug_empty["candidate_ys"] = ys_all
        debug_empty["unassigned_xs"] = xs_all
        debug_empty["unassigned_ys"] = ys_all
        return [], debug_empty

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        80,
        0.5,
    )

    cv2.setRNGSeed(int(random_seed))
    _, _, centers = cv2.kmeans(
        points_fit,
        n_components,
        None,
        criteria,
        int(max(1, attempts)),
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(np.float32)

    points_assign = np.column_stack([xs_fit_full, ys_fit_full]).astype(np.float32)
    d2 = ((points_assign[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels_raw = np.argmin(d2, axis=1).astype(np.int32)

    comps_raw = []
    for k in range(n_components):
        idx = labels_raw == k
        n_pts = int(np.count_nonzero(idx))
        if n_pts < min_events_per_cluster:
            continue

        xk = xs_fit_full[idx]
        yk = ys_fit_full[idx]
        bbox_x, bbox_y, bbox_w, bbox_h = _clip_bbox_from_points(
            xk,
            yk,
            width=width,
            height=height,
            bbox_pad=bbox_pad,
        )

        comps_raw.append({
            "area": n_pts,
            "centroid_x": float(np.mean(xk)),
            "centroid_y": float(np.mean(yk)),
            "bbox_x": int(bbox_x),
            "bbox_y": int(bbox_y),
            "bbox_w": int(bbox_w),
            "bbox_h": int(bbox_h),
            "source": "event_cluster",
            "_raw_cluster_id": int(k),
        })

    comps_raw.sort(key=lambda c: (c["centroid_x"], c["centroid_y"]))

    label_map = {}
    components = []
    for new_id, comp in enumerate(comps_raw):
        raw_id = int(comp.pop("_raw_cluster_id"))
        label_map[raw_id] = int(new_id)
        comp["cluster_id"] = int(new_id)
        components.append(comp)


    mapped_labels = np.full(len(labels_raw), -1, dtype=np.int32)
    for raw_id, new_id in label_map.items():
        mapped_labels[labels_raw == raw_id] = new_id

    valid_assigned = mapped_labels >= 0
    assigned_xs = xs_fit_full[valid_assigned]
    assigned_ys = ys_fit_full[valid_assigned]
    assigned_labels = mapped_labels[valid_assigned]

    failed_fit = np.ones(len(xs_fit_full), dtype=bool)
    failed_fit[valid_assigned] = False
    unassigned_xs = xs_fit_full[failed_fit].astype(np.float32)
    unassigned_ys = ys_fit_full[failed_fit].astype(np.float32)

    centers_sorted = np.array(
        [[c["centroid_x"], c["centroid_y"]] for c in components],
        dtype=np.float32,
    )

    debug = {
        "candidate_xs": xs_all.astype(np.float32),
        "candidate_ys": ys_all.astype(np.float32),
        "assigned_xs": assigned_xs.astype(np.float32),
        "assigned_ys": assigned_ys.astype(np.float32),
        "assigned_labels": assigned_labels.astype(np.int32),
        "unassigned_xs": unassigned_xs,
        "unassigned_ys": unassigned_ys,
        "centers": centers_sorted,
    }

    return components, debug
