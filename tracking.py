import numpy as np


def initialize_fixed_tracks(components, n_propellers=2, bootstrap_sort="x"):
    if n_propellers < 1:
        raise ValueError("n_propellers must be >= 1")
    if len(components) < n_propellers:
        return None

    selected = list(components[:n_propellers])
    if bootstrap_sort == "x":
        selected.sort(key=lambda c: (c["centroid_x"], c["centroid_y"]))
    elif bootstrap_sort == "y":
        selected.sort(key=lambda c: (c["centroid_y"], c["centroid_x"]))
    else:
        raise ValueError("bootstrap_sort must be one of: 'x', 'y'")

    tracks = {}
    for tid, comp in enumerate(selected):
        tracks[tid] = {
            "track_id": tid,
            "centroid_x": float(comp["centroid_x"]),
            "centroid_y": float(comp["centroid_y"]),
            "label_centroid_x": float(comp["centroid_x"]),
            "label_centroid_y": float(comp["centroid_y"]),
            "last_component": dict(comp),
            "missed_windows": 0,
            "prev_rpm": np.nan,
            "locked_sign": None,
            "last_full_search_t0_us": None,
            "initialized": True,
        }
    return tracks


def assign_components_to_fixed_tracks(components, tracks, max_dx_per_window=20.0, max_dy_per_window=20.0, max_distance_per_window=None):
    track_ids = sorted(tracks.keys())
    if len(track_ids) == 0:
        return {}

    candidate_lists = {}
    unmatched_costs = {}

    for tid in track_ids:
        st = tracks[tid]
        miss = int(st.get("missed_windows", 0))
        scale = miss + 1
        max_dx_eff = np.inf if max_dx_per_window is None else float(max_dx_per_window) * scale
        max_dy_eff = np.inf if max_dy_per_window is None else float(max_dy_per_window) * scale

        if max_distance_per_window is None:
            max_dist_eff = float(np.hypot(max_dx_eff, max_dy_eff)) if np.isfinite(max_dx_eff) and np.isfinite(max_dy_eff) else np.inf
        else:
            max_dist_eff = float(max_distance_per_window) * scale

        unmatched_costs[tid] = max_dist_eff + 1.0 if np.isfinite(max_dist_eff) else 50.0
        cands = []
        for ci, comp in enumerate(components):
            dx = float(comp["centroid_x"] - st["centroid_x"])
            dy = float(comp["centroid_y"] - st["centroid_y"])
            dist = float(np.hypot(dx, dy))
            if abs(dx) > max_dx_eff or abs(dy) > max_dy_eff or dist > max_dist_eff:
                continue
            cands.append((ci, dist))
        candidate_lists[tid] = cands

    best_cost = np.inf
    best_assignment = {tid: None for tid in track_ids}

    def dfs(i, used_components, current_cost, current_assignment):
        nonlocal best_cost, best_assignment
        if current_cost >= best_cost:
            return
        if i == len(track_ids):
            best_cost = current_cost
            best_assignment = current_assignment.copy()
            return

        tid = track_ids[i]
        current_assignment[tid] = None
        dfs(i + 1, used_components, current_cost + unmatched_costs[tid], current_assignment)

        for ci, cost in candidate_lists[tid]:
            if ci in used_components:
                continue
            used_components.add(ci)
            current_assignment[tid] = ci
            dfs(i + 1, used_components, current_cost + cost, current_assignment)
            used_components.remove(ci)

    dfs(0, set(), 0.0, {})
    return best_assignment


def update_fixed_tracks(components, tracks, assignment):
    matched_components = {}
    for tid, st in tracks.items():
        ci = assignment.get(tid, None)
        if ci is None:
            st["missed_windows"] = int(st.get("missed_windows", 0)) + 1
            continue

        comp = components[ci]
        st["centroid_x"] = float(comp["centroid_x"])
        st["centroid_y"] = float(comp["centroid_y"])
        st["last_component"] = dict(comp)
        st["missed_windows"] = 0
        st["initialized"] = True
        matched_components[tid] = comp

    return matched_components, tracks
