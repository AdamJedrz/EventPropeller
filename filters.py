import numpy as np
import cv2


def keep_large_inner_bin_components(active_bins_mask, min_component_area=20, remove_border_components=True):
    mask = active_bins_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = np.zeros_like(mask, dtype=np.uint8)
    height, width = mask.shape

    for lbl in range(1, num_labels):
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]

        touches_border = (x == 0) or (y == 0) or (x + w >= width) or (y + h >= height)

        if area < min_component_area:
            continue
        if remove_border_components and touches_border:
            continue

        out[labels == lbl] = 1

    return out.astype(bool)


def apply_bin_morphology(active_bins_mask, mode="none", kernel_size=3, iterations=0):
    mode = str(mode or "none").lower()
    iterations = int(iterations or 0)
    kernel_size = int(kernel_size or 1)

    if mode in ("none", "off", "false") or iterations <= 0 or kernel_size <= 1:
        return active_bins_mask.astype(bool)

    if mode not in ("erode", "open", "opening"):
        raise ValueError("density_bin_morph_mode musi być jednym z: 'none', 'erode', 'open'")

    mask = active_bins_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if mode == "erode":
        out = cv2.erode(mask, kernel, iterations=iterations)
    else:
        out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    return (out > 0)


def keep_mask_from_active_bins(xs, ys, image_shape, bin_size, active_bins):
    height, width = image_shape
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    active_bins = np.asarray(active_bins).astype(bool)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    keep_mask = np.zeros(len(xs), dtype=bool)

    if not np.any(valid) or active_bins.size == 0:
        return keep_mask

    bx = xs[valid] // int(bin_size)
    by = ys[valid] // int(bin_size)

    in_bins = (by >= 0) & (by < active_bins.shape[0]) & (bx >= 0) & (bx < active_bins.shape[1])
    keep_valid = np.zeros(np.count_nonzero(valid), dtype=bool)
    keep_valid[in_bins] = active_bins[by[in_bins], bx[in_bins]]
    keep_mask[valid] = keep_valid

    return keep_mask


def density_filter_events(
    xs,
    ys,
    image_shape,
    bin_size=7,
    min_events_in_bin=3,
    min_component_area=18,
    remove_border_components=True,
    return_active_bins=False,
):
    height, width = image_shape
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs_valid = xs[valid]
    ys_valid = ys[valid]

    n_bins_x = (width + bin_size - 1) // bin_size
    n_bins_y = (height + bin_size - 1) // bin_size

    bx = xs_valid // bin_size
    by = ys_valid // bin_size

    counts = np.zeros((n_bins_y, n_bins_x), dtype=np.int32)
    np.add.at(counts, (by, bx), 1)

    active_bins = counts >= min_events_in_bin
    active_bins = keep_large_inner_bin_components(
        active_bins,
        min_component_area=min_component_area,
        remove_border_components=remove_border_components,
    )

    keep_mask = keep_mask_from_active_bins(xs, ys, image_shape, bin_size, active_bins)

    if return_active_bins:
        return keep_mask, active_bins
    return keep_mask
