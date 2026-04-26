import cv2


def extract_components(mask_u8, bbox_pad=4, max_components=None):
    binary = (mask_u8 > 0).astype("uint8")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = []
    height, width = binary.shape

    for lbl in range(1, num_labels):
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        cx, cy = centroids[lbl]

        x0 = max(0, x - bbox_pad)
        y0 = max(0, y - bbox_pad)
        x1 = min(width - 1, x + w - 1 + bbox_pad)
        y1 = min(height - 1, y + h - 1 + bbox_pad)

        comps.append({
            "area": int(area),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "bbox_x": int(x0),
            "bbox_y": int(y0),
            "bbox_w": int(x1 - x0 + 1),
            "bbox_h": int(y1 - y0 + 1),
        })

    comps.sort(key=lambda c: c["area"], reverse=True)
    if max_components is not None:
        comps = comps[:max_components]

    for i, comp in enumerate(comps):
        comp["cluster_id"] = i

    return comps
