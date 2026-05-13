import numpy as np
import cv2

from polarity import ensure_signed_polarity, select_events_by_polarity
from filters import density_filter_events
from motion_compensation import crop_events_to_component_bbox, warp_events_about_centroid, build_local_histogram


def add_title_bar(img, title, bar_height=32):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    height, width = img.shape[:2]
    out = np.zeros((height + bar_height, width, 3), dtype=np.uint8)
    out[bar_height:] = img

    cv2.putText(out, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_for_screen(img, max_width=1400, max_height=900):
    height, width = img.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return img

    return cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def draw_events_gray(xs, ys, ps, image_shape):
    height, width = image_shape
    img = np.full((height, width), 127, dtype=np.uint8)

    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    ps = ensure_signed_polarity(ps)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    ps = ps[valid]

    img[ys[ps > 0], xs[ps > 0]] = 255
    img[ys[ps < 0], xs[ps < 0]] = 0

    return img


def draw_mask_yellow(mask_u8):
    img = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    img[mask_u8 > 0] = (0, 255, 255)
    return img


def draw_components_overlay(base_img, components, color=(0, 255, 0)):
    img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR) if base_img.ndim == 2 else base_img.copy()

    for comp in components:
        x, y = comp["bbox_x"], comp["bbox_y"]
        w, h = comp["bbox_w"], comp["bbox_h"]
        cx = int(round(comp["centroid_x"]))
        cy = int(round(comp["centroid_y"]))
        cid = comp.get("track_id", comp.get("cluster_id", "?"))

        opt_x = comp.get("optimized_center_x", comp["centroid_x"])
        opt_y = comp.get("optimized_center_y", comp["centroid_y"])
        ox = int(round(opt_x))
        oy = int(round(opt_y))

        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, 2)

        # centroid z maski/segmentacji
        cv2.circle(img, (cx, cy), 7, (255, 255, 255), -1)
        cv2.circle(img, (cx, cy), 9, (0, 0, 255), 2)

        # optymalny środek obrotu użyty do warpingu
        cv2.drawMarker(img, (ox, oy), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.circle(img, (ox, oy), 6, (255, 255, 0), -1)
        cv2.circle(img, (ox, oy), 9, (255, 0, 0), 2)

        cv2.putText(img, f"C{cid}", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        if "optimized_center_dx" in comp and "optimized_center_dy" in comp:
            txt = f"d=({comp['optimized_center_dx']:.0f},{comp['optimized_center_dy']:.0f})"
            cv2.putText(img, txt, (x, min(img.shape[0] - 8, y + h + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        if "optimized_q" in comp:
            txt_q = f"q={comp['optimized_q']:.2f}"
            cv2.putText(img, txt_q, (x, min(img.shape[0] - 8, y + h + 32)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    return img


def components_from_matched(matched_components):
    out = []
    for tid, comp in (matched_components or {}).items():
        if comp is None:
            continue
        c = dict(comp)
        c["track_id"] = tid
        out.append(c)
    return out


def first_bundle_filtered(first_bundle, image_shape, components, polarity_mode, filter_params):
    xs, ys, ps, _ = select_events_by_polarity(first_bundle["events"], polarity_mode)
    keep_mask = density_filter_events(xs, ys, image_shape=image_shape, **(filter_params or {}))
    img = draw_events_gray(xs[keep_mask], ys[keep_mask], ps[keep_mask], image_shape)
    return draw_components_overlay(img, components)


def hist_to_colormap(full_hist, image_shape):
    if full_hist.max() <= 0:
        return np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    vis = np.log1p(full_hist)
    vis = (255.0 * vis / vis.max()).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def first_bundle_warp_and_histograms(
    first_bundle,
    image_shape,
    matched_components,
    rpm_by_track,
    polarity_mode,
    reference_time_fractions,
):
    components = components_from_matched(matched_components)

    if not matched_components:
        empty = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        return empty, []

    xs_all, ys_all, ps_all, ts_all = select_events_by_polarity(first_bundle["events"], polarity_mode)

    t0 = float(first_bundle["t0"])
    t1 = float(first_bundle["t1"])
    reference_time_fractions = tuple(reference_time_fractions)
    reference_times = [t0 + float(frac) * (t1 - t0) for frac in reference_time_fractions]

    middle_idx = int(np.argmin([abs(float(frac) - 0.5) for frac in reference_time_fractions]))

    global_hists = [np.zeros(image_shape, dtype=np.float32) for _ in reference_times]
    all_xw_middle = []
    all_yw_middle = []
    all_pw_middle = []

    for tid, comp in matched_components.items():
        if comp is None:
            continue

        rpm = comp.get("preview_rpm", rpm_by_track.get(tid, 0.0))
        if rpm is None or np.isnan(rpm):
            rpm = 0.0

        center_x = comp.get("optimized_center_x", comp["centroid_x"])
        center_y = comp.get("optimized_center_y", comp["centroid_y"])
        q = comp.get("optimized_q", 1.0)
        q_axis_angle_deg = comp.get("q_axis_angle_deg", 0.0)

        xs, ys, ps, ts = crop_events_to_component_bbox(xs_all, ys_all, ps_all, ts_all, comp)
        if len(xs) == 0:
            continue

        omega = float(rpm) * 2.0 * np.pi / 60.0

        for ref_idx, t_ref_us in enumerate(reference_times):
            xw, yw = warp_events_about_centroid(
                xs,
                ys,
                ts,
                center_x,
                center_y,
                omega,
                t_ref_us,
                q=q,
                q_axis_angle_rad=np.deg2rad(float(q_axis_angle_deg)),
            )
            hist = build_local_histogram(xw, yw, comp)

            x0 = comp["bbox_x"]
            y0 = comp["bbox_y"]
            h, w = hist.shape
            y1 = min(image_shape[0], y0 + h)
            x1 = min(image_shape[1], x0 + w)
            global_hists[ref_idx][y0:y1, x0:x1] += hist[: y1 - y0, : x1 - x0].astype(np.float32)

            if ref_idx == middle_idx:
                xi = np.rint(xw).astype(np.int32)
                yi = np.rint(yw).astype(np.int32)
                valid = (xi >= 0) & (xi < image_shape[1]) & (yi >= 0) & (yi < image_shape[0])
                if np.any(valid):
                    all_xw_middle.append(xw[valid])
                    all_yw_middle.append(yw[valid])
                    all_pw_middle.append(ps[valid])

    if len(all_xw_middle) > 0:
        warped = draw_events_gray(
            np.concatenate(all_xw_middle),
            np.concatenate(all_yw_middle),
            np.concatenate(all_pw_middle),
            image_shape,
        )
    else:
        warped = np.full(image_shape, 127, dtype=np.uint8)

    warped_overlay = draw_components_overlay(warped, components)

    hist_overlays = []
    for frac, hist in zip(reference_time_fractions, global_hists):
        hist_img = hist_to_colormap(hist, image_shape)
        hist_overlays.append((frac, draw_components_overlay(hist_img, components)))

    return warped_overlay, hist_overlays


def build_preview_images(
    window_dict,
    process_result,
    image_shape,
    polarity_mode,
    filter_params=None,
    matched_components=None,
    rpm_by_track=None,
    reference_time_fractions=(0.0, 0.5, 1.0),
):
    matched_components = matched_components or {}
    rpm_by_track = rpm_by_track or {}

    xs, ys, ps, _ = select_events_by_polarity(window_dict["events"], polarity_mode)
    raw_img = draw_events_gray(xs, ys, ps, image_shape)

    keep_mask = density_filter_events(xs, ys, image_shape=image_shape, **(filter_params or {}))
    filtered_img = draw_events_gray(xs[keep_mask], ys[keep_mask], ps[keep_mask], image_shape)
    filtered_overlay = draw_components_overlay(filtered_img, process_result["components"])
    closed_overlay = draw_components_overlay(draw_mask_yellow(process_result["closed_mask"]), process_result["components"])

    first_bundle = window_dict["bundles"][0]
    components_for_first = components_from_matched(matched_components) if matched_components else process_result["components"]
    first_filtered = first_bundle_filtered(first_bundle, image_shape, components_for_first, polarity_mode, filter_params)

    if not matched_components:
        matched_components = {i: comp for i, comp in enumerate(process_result["components"])}
        for i in matched_components:
            rpm_by_track.setdefault(i, 0.0)

    first_warped, first_histograms = first_bundle_warp_and_histograms(
        first_bundle=first_bundle,
        image_shape=image_shape,
        matched_components=matched_components,
        rpm_by_track=rpm_by_track,
        polarity_mode=polarity_mode,
        reference_time_fractions=reference_time_fractions,
    )

    return {
        "raw_img": raw_img,
        "filtered_overlay": filtered_overlay,
        "closed_overlay": closed_overlay,
        "first_bundle_filtered_overlay": first_filtered,
        "first_bundle_warped_overlay": first_warped,
        "first_bundle_histograms": first_histograms,
    }


def show_preview(
    window_dict,
    process_result,
    image_shape,
    polarity_mode,
    wait_ms=1,
    filter_params=None,
    matched_components=None,
    rpm_by_track=None,
    reference_time_fractions=(0.0, 0.5, 1.0),
):
    imgs = build_preview_images(
        window_dict,
        process_result,
        image_shape,
        polarity_mode,
        filter_params=filter_params,
        matched_components=matched_components,
        rpm_by_track=rpm_by_track,
        reference_time_fractions=reference_time_fractions,
    )

    cv2.imshow("1_All_events_window", resize_for_screen(add_title_bar(imgs["raw_img"], f"1. Wszystkie eventy | window={process_result['window_idx']} | N={process_result['n_events_raw']}")))
    cv2.imshow("2_Filtered_window", resize_for_screen(add_title_bar(imgs["filtered_overlay"], f"2. Po filtracji + komponenty | N={process_result['n_events_filtered']} | C={len(process_result['components'])}")))
    cv2.imshow("3_Closed_mask_window", resize_for_screen(add_title_bar(imgs["closed_overlay"], f"3. Maska ROI + komponenty | C={len(process_result['components'])}")))
    cv2.imshow("4_First_bundle_filtered", resize_for_screen(add_title_bar(imgs["first_bundle_filtered_overlay"], "4. Pierwszy bundle po filtracji")))
    cv2.imshow("5_First_bundle_warped_middle", resize_for_screen(add_title_bar(imgs["first_bundle_warped_overlay"], "5. Pierwszy bundle po warpingu do ref ~0.5")))

    for i, (frac, hist_img) in enumerate(imgs["first_bundle_histograms"], start=1):
        cv2.imshow(
            f"6{i}_First_bundle_hist_ref_{frac:.2f}",
            resize_for_screen(add_title_bar(hist_img, f"6.{i} Histogram warped events | ref fraction={frac:.2f}")),
        )

    key = cv2.waitKey(wait_ms) & 0xFF
    return key != 27


def close_preview_windows():
    cv2.destroyAllWindows()
