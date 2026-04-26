import numpy as np
import cv2


def events_to_binary_mask(xs, ys, image_shape):
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    mask[ys[valid], xs[valid]] = 255

    return mask


def remove_small_and_border_components(mask_u8, min_area_px=1000, remove_border_components=True):
    binary = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    out = np.zeros_like(binary, dtype=np.uint8)
    height, width = binary.shape

    for lbl in range(1, num_labels):
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]

        touches_border = (x == 0) or (y == 0) or (x + w >= width) or (y + h >= height)

        if area < min_area_px:
            continue
        if remove_border_components and touches_border:
            continue

        out[labels == lbl] = 1

    return (out * 255).astype(np.uint8)


def fill_binary_holes(mask_u8):
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    height, width = mask_u8.shape

    flood = mask_u8.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)

    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_u8, holes)


def build_closed_mask(
    xs,
    ys,
    image_shape,
    dilate_kernel_size=5,
    dilate_iterations=1,
    close_kernel_size=9,
    close_iterations=1,
    fill_holes=True,
    min_component_area_px=3000,
    remove_border_components=True,
):
    """
    binary -> dilate -> close -> remove small/border -> fill holes -> remove small/border.
    """
    mask = events_to_binary_mask(xs, ys, image_shape)

    if dilate_kernel_size > 1 and dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)

    if close_kernel_size > 1 and close_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    mask = remove_small_and_border_components(
        mask,
        min_area_px=min_component_area_px,
        remove_border_components=remove_border_components,
    )

    if fill_holes:
        mask = fill_binary_holes(mask)

    return remove_small_and_border_components(
        mask,
        min_area_px=min_component_area_px,
        remove_border_components=remove_border_components,
    )
