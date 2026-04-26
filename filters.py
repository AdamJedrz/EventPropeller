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

    keep_mask = np.zeros(len(xs), dtype=bool)
    keep_mask[valid] = active_bins[by, bx]

    if return_active_bins:
        return keep_mask, active_bins
    return keep_mask
