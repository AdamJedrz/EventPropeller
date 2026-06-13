from pathlib import Path
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

INPUT_DIR = Path("outputs/maneuver_org")
OUT_DIR_DEFAULT = Path("maneuver_model")

DATASETS = {
    "climb_ir": {
        "file": INPUT_DIR / "rpm_results_climb_eventpro.csv",
        "segments": [
            (0, 1260, "hover"),
            (1260, 1500, "climb"),
            (1500, 3050, "hover"),
            (3050, 3700, "descent"),
            (3700, None, "hover"),
        ],
    },
    "descent_ir": {
        "file": INPUT_DIR / "rpm_results_descent_eventpro.csv",
        "segments": [
            (1000, 1900, "descent"),
            (1900, 2200, "climb"),
            (2200, 2500, "yaw"),
            (2500, 3300, "hover"),
            (3300, 3600, "climb"),
            (3600, None, "hover"),
        ],
    },
    "hover_ir": {
        "file": INPUT_DIR / "rpm_results_hover_eventpro.csv",
        "segments": [(0, None, "hover")],
    },
    "yaw_ir": {
        "file": INPUT_DIR / "rpm_results_yaw_eventpro.csv",
        "segments": [
            (0, 1400, "hover"),
            (1400, 1600, "yaw"),
            (1600, 15050, "hover"),
            (15050, 15260, "yaw"),
            (15260, None, "hover"),
        ],
    },
    "roll_pitch_ir": {
        "file": INPUT_DIR / "rpm_results_roll_eventpro.csv",
        "segments": [
            (0, 3480, "hover"),
            (4000, 4640, "roll_pitch"),
            (5700, 6240, "roll_pitch"),
            (7160, 7720, "roll_pitch"),
            (8700, 9280, "roll_pitch"),
        ],
    },
}


CROP_ORDER = ["climb_ir", "descent_ir", "hover_ir", "yaw_ir", "roll_pitch_ir"]

CROP_INTERVALS = {
    "climb_ir": (1100, 1800),
    "descent_ir": (1200, 2000),
    "hover_ir": (0, 2000),
    "yaw_ir": (1000, 2000),
    "roll_pitch_ir": (5700, 6000),
}

CROP_TITLES = {
    "climb_ir": "climb",
    "descent_ir": "descent",
    "hover_ir": "hover",
    "yaw_ir": "yaw",
    "roll_pitch_ir": "roll/pitch",
}


WINDOW_MS = 120.0
STEP_MS = 60.0
MIN_SAMPLES_IN_WINDOW = 2

TEST_SIZE = 0.4

SVM_C = 10.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"

RANDOM_STATE = 42

BALANCE_TRAIN_HOVER = True
HOVER_RATIO_TO_MAX_OTHER = 1.5

AUGMENT_TRAIN = True
AUGMENT_TRAIN_NOISE_FRAC = 0.015

AUGMENT_TEST_FOR_METRICS = True
BALANCED_TEST_TARGET_PER_CLASS = 80
AUGMENT_TEST_NOISE_FRAC = 0.015

LABEL_ORDER = ["climb", "descent", "hover", "roll_pitch", "yaw"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABEL_ORDER)}
RPM_COLS = ["c1", "c2", "c3", "c4"]


def load_rpm_csv(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required = ["time"] + RPM_COLS
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Brakuje kolumny '{col}' w pliku {csv_path}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time"]).copy()
    df = df.sort_values("time").reset_index(drop=True)

    df[RPM_COLS] = df[RPM_COLS].interpolate(limit_direction="both")
    df = df.dropna(subset=RPM_COLS).reset_index(drop=True)

    return df


def end_or_max(end_ms, df):
    if end_ms is None:
        return float(df["time"].max())
    return float(end_ms)


def safe_slope(t_ms, y):
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t_ms) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return 0.0

    t = t_ms[valid] - t_ms[valid][0]
    yv = y[valid]

    if np.nanmax(t) <= 0:
        return 0.0

    try:
        return float(np.polyfit(t / 1000.0, yv, 1)[0])
    except Exception:
        return 0.0


def stats_for_signal(prefix, t_ms, y):
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t_ms) & np.isfinite(y)

    if np.count_nonzero(valid) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_start": 0.0,
            f"{prefix}_end": 0.0,
            f"{prefix}_delta": 0.0,
            f"{prefix}_slope": 0.0,
            f"{prefix}_diff_mean": 0.0,
            f"{prefix}_diff_std": 0.0,
            f"{prefix}_diff_abs_mean": 0.0,
        }

    tv = t_ms[valid]
    yv = y[valid]

    out = {
        f"{prefix}_mean": float(np.mean(yv)),
        f"{prefix}_std": float(np.std(yv)),
        f"{prefix}_min": float(np.min(yv)),
        f"{prefix}_max": float(np.max(yv)),
        f"{prefix}_median": float(np.median(yv)),
        f"{prefix}_range": float(np.max(yv) - np.min(yv)),
        f"{prefix}_start": float(yv[0]),
        f"{prefix}_end": float(yv[-1]),
        f"{prefix}_delta": float(yv[-1] - yv[0]),
        f"{prefix}_slope": safe_slope(tv, yv),
    }

    if len(yv) >= 2:
        dy = np.diff(yv)
        out[f"{prefix}_diff_mean"] = float(np.mean(dy))
        out[f"{prefix}_diff_std"] = float(np.std(dy))
        out[f"{prefix}_diff_abs_mean"] = float(np.mean(np.abs(dy)))
    else:
        out[f"{prefix}_diff_mean"] = 0.0
        out[f"{prefix}_diff_std"] = 0.0
        out[f"{prefix}_diff_abs_mean"] = 0.0

    return out


def extract_features_from_window(win_df):
    t = win_df["time"].to_numpy(dtype=float)

    rpm_signed = win_df[RPM_COLS].to_numpy(dtype=float)
    rpm_abs = np.abs(rpm_signed)
    rpm_sq = rpm_abs ** 2
    rpm_signed_sq = np.sign(rpm_signed) * (np.abs(rpm_signed) ** 2)

    features = {}

    for i in range(4):
        features.update(stats_for_signal(f"abs_c{i+1}", t, rpm_abs[:, i]))
        features.update(stats_for_signal(f"signed_c{i+1}", t, rpm_signed[:, i]))
        features.update(stats_for_signal(f"sq_c{i+1}", t, rpm_sq[:, i]))
        features.update(stats_for_signal(f"signed_sq_c{i+1}", t, rpm_signed_sq[:, i]))

    mean_rpm = np.mean(rpm_abs, axis=1)
    spread_rpm = np.std(rpm_abs, axis=1)
    thrust_proxy = np.sum(rpm_sq, axis=1)

    features.update(stats_for_signal("mean_rpm", t, mean_rpm))
    features.update(stats_for_signal("spread_rpm", t, spread_rpm))
    features.update(stats_for_signal("thrust_proxy", t, thrust_proxy))

    eps = 1e-9
    features.update(stats_for_signal("spread_rel", t, spread_rpm / (mean_rpm + eps)))

    for i, j in combinations(range(4), 2):
        diff_abs = rpm_abs[:, i] - rpm_abs[:, j]
        diff_sq = rpm_sq[:, i] - rpm_sq[:, j]
        diff_signed = rpm_signed[:, i] - rpm_signed[:, j]
        diff_signed_sq = rpm_signed_sq[:, i] - rpm_signed_sq[:, j]

        features.update(stats_for_signal(f"diff_abs_c{i+1}_c{j+1}", t, diff_abs))
        features.update(stats_for_signal(f"diff_sq_c{i+1}_c{j+1}", t, diff_sq))
        features.update(stats_for_signal(f"diff_signed_c{i+1}_c{j+1}", t, diff_signed))
        features.update(stats_for_signal(f"diff_signed_sq_c{i+1}_c{j+1}", t, diff_signed_sq))

    p14_23 = (rpm_sq[:, 0] + rpm_sq[:, 3]) - (rpm_sq[:, 1] + rpm_sq[:, 2])
    p12_34 = (rpm_sq[:, 0] + rpm_sq[:, 1]) - (rpm_sq[:, 2] + rpm_sq[:, 3])
    p13_24 = (rpm_sq[:, 0] + rpm_sq[:, 2]) - (rpm_sq[:, 1] + rpm_sq[:, 3])

    features.update(stats_for_signal("pair_14_minus_23", t, p14_23))
    features.update(stats_for_signal("pair_12_minus_34", t, p12_34))
    features.update(stats_for_signal("pair_13_minus_24", t, p13_24))

    features.update(stats_for_signal("abs_pair_14_minus_23", t, np.abs(p14_23)))
    features.update(stats_for_signal("abs_pair_12_minus_34", t, np.abs(p12_34)))
    features.update(stats_for_signal("abs_pair_13_minus_24", t, np.abs(p13_24)))

    prop_mean_abs = np.mean(rpm_abs, axis=0)
    prop_mean_sq = np.mean(rpm_sq, axis=0)

    features["prop_mean_abs_std"] = float(np.std(prop_mean_abs))
    features["prop_mean_sq_std"] = float(np.std(prop_mean_sq))
    features["prop_mean_abs_range"] = float(np.max(prop_mean_abs) - np.min(prop_mean_abs))
    features["prop_mean_sq_range"] = float(np.max(prop_mean_sq) - np.min(prop_mean_sq))

    return features


def split_for_window(win_start, win_end, seg_start, seg_end, test_size):
    seg_len = float(seg_end - seg_start)
    split_t = float(seg_start + (1.0 - test_size) * seg_len)

    if win_end <= split_t:
        return "train"
    if win_start >= split_t:
        return "test"
    return "ignore_boundary"


def build_windows_for_dataset(recording_name, raw_df, segments, window_ms, step_ms, min_samples, test_size):
    rows = []

    for segment_id, (seg_start, seg_end, label) in enumerate(segments):
        seg_start = float(seg_start)
        seg_end_eff = end_or_max(seg_end, raw_df)

        start = seg_start

        while start + window_ms <= seg_end_eff + 1e-9:
            end = start + window_ms

            win_df = raw_df[
                (raw_df["time"] >= start) &
                (raw_df["time"] <= end)
            ].copy()

            if len(win_df) >= min_samples:
                feats = extract_features_from_window(win_df)

                row_split = split_for_window(
                    win_start=start,
                    win_end=end,
                    seg_start=seg_start,
                    seg_end=seg_end_eff,
                    test_size=test_size,
                )

                row = {
                    "recording": recording_name,
                    "label": label,
                    "segment_id": int(segment_id),
                    "segment_start_ms": float(seg_start),
                    "segment_end_ms": float(seg_end_eff),
                    "window_start_ms": float(start),
                    "window_end_ms": float(end),
                    "window_center_ms": float(0.5 * (start + end)),
                    "split": row_split,
                    **feats,
                }

                rows.append(row)

            start += step_ms

    return pd.DataFrame(rows)

def downsample_hover_train(train_df, hover_ratio_to_max_other=1.5, random_state=42):
    if "hover" not in set(train_df["label"]):
        return train_df

    counts = train_df["label"].value_counts()
    other_counts = counts.drop(labels=["hover"], errors="ignore")

    if len(other_counts) == 0:
        return train_df

    max_other = int(other_counts.max())
    max_hover = int(max(1, round(hover_ratio_to_max_other * max_other)))

    parts = []

    for label, part in train_df.groupby("label"):
        if label == "hover" and len(part) > max_hover:
            part = part.sample(n=max_hover, random_state=random_state)
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return out


def augment_features_to_target(X, y, target_per_class, classes, noise_frac, random_state, allow_downsample=False):
    X = X.copy().reset_index(drop=True)
    y = pd.Series(y).astype(str).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    feature_std = X.std(axis=0).replace(0, 1.0)

    X_parts = []
    y_parts = []

    for cls in classes:
        mask = (y == cls)
        X_cls = X.loc[mask].copy()
        n = len(X_cls)

        if n == 0:
            continue

        if allow_downsample and n > target_per_class:
            X_cls = X_cls.sample(n=target_per_class, random_state=random_state)
            n = len(X_cls)

        X_parts.append(X_cls.reset_index(drop=True))
        y_parts.append(pd.Series([cls] * n))

        need = int(target_per_class - n)

        if need > 0:
            sample_idx = rng.choice(n, size=need, replace=True)
            base = X_cls.reset_index(drop=True).iloc[sample_idx].reset_index(drop=True)

            noise = rng.normal(
                loc=0.0,
                scale=noise_frac,
                size=base.shape,
            )

            X_aug = base + noise * feature_std.to_numpy()
            y_aug = pd.Series([cls] * need)

            X_parts.append(X_aug.reset_index(drop=True))
            y_parts.append(y_aug.reset_index(drop=True))

    X_out = pd.concat(X_parts, ignore_index=True)
    y_out = pd.concat(y_parts, ignore_index=True)

    perm = rng.permutation(len(X_out))
    X_out = X_out.iloc[perm].reset_index(drop=True)
    y_out = y_out.iloc[perm].reset_index(drop=True)

    return X_out, y_out


def _clip_interval(a, b, t0, t1):
    a = max(float(a), float(t0))
    b = min(float(b), float(t1))
    if b <= a:
        return None
    return a, b


def gt_segments_to_intervals(segments, t_max, t0=None, t1=None):
    out = []

    for start, end, label in segments:
        a = float(start)
        b = float(t_max if end is None else end)

        if t0 is not None and t1 is not None:
            clipped = _clip_interval(a, b, t0, t1)
            if clipped is None:
                continue
            a, b = clipped

        out.append((a, b, str(label)))

    return out


def prediction_rows_to_intervals(pred_df, label_col="pred_label", t0=None, t1=None):
    pred_df = pred_df.sort_values("window_center_ms").reset_index(drop=True)

    if len(pred_df) == 0:
        return []

    centers = pred_df["window_center_ms"].to_numpy(dtype=float)
    intervals = []

    for i, row in pred_df.iterrows():
        if i == 0:
            a = float(row["window_start_ms"])
        else:
            a = 0.5 * (centers[i - 1] + centers[i])

        if i == len(pred_df) - 1:
            b = float(row["window_end_ms"])
        else:
            b = 0.5 * (centers[i] + centers[i + 1])

        if t0 is not None and t1 is not None:
            clipped = _clip_interval(a, b, t0, t1)
            if clipped is None:
                continue
            a, b = clipped

        intervals.append((a, b, str(row[label_col])))

    return intervals


def plot_intervals_as_steps(ax, intervals, label_to_id, label_name, color, linewidth, y_offset=0.0, alpha=1.0, zorder=2):
    intervals = sorted(intervals, key=lambda x: (float(x[0]), float(x[1])))

    xs = []
    ys = []
    prev_y = None

    for start, end, label in intervals:
        if label not in label_to_id:
            continue

        start = float(start)
        end = float(end)
        y = float(label_to_id[label]) + y_offset

        if prev_y is not None:
            xs.append(start)
            ys.append(prev_y)
            if y != prev_y:
                xs.append(start)
                ys.append(y)

        xs.extend([start, end])
        ys.extend([y, y])
        prev_y = y

    if xs:
        ax.plot(xs, ys, label=label_name, color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="butt", zorder=zorder)


def plot_rpm_channels(ax, raw_df, t0=None, t1=None, y_min=None, y_max=None, show_legend=False):
    if t0 is not None and t1 is not None:
        df = raw_df[(raw_df["time"] >= t0) & (raw_df["time"] <= t1)].copy()
    else:
        df = raw_df.copy()

    if len(df) == 0:
        return

    rpm_colors = {"c1": "tab:blue", "c2": "tab:orange", "c3": "tab:green", "c4": "tab:red"}

    for col in RPM_COLS:
        ax.plot(df["time"], np.abs(df[col]), label=col, linewidth=1.4, color=rpm_colors.get(col, None))

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.35)

    if show_legend:
        ax.legend(fontsize=8, loc="upper right")


def save_confusion_matrix_plot(cm, labels, out_path, title, values_format):
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format=values_format, cmap="Blues", colorbar=True)
    ax.set_title(title)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Ground truth")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_recording_prediction(recording_name, raw_df, pred_df, segments, out_path):
    t_max = float(raw_df["time"].max())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.3]})

    plot_rpm_channels(ax=ax1, raw_df=raw_df, show_legend=True)
    ax1.set_title(f"RPM i klasyfikacja manewru: {recording_name}")
    ax1.set_ylabel("RPM")

    gt_intervals = gt_segments_to_intervals(segments, t_max=t_max)
    pred_intervals = prediction_rows_to_intervals(pred_df, label_col="pred_label")

    plot_intervals_as_steps(ax2, gt_intervals, LABEL_TO_INT, label_name="GT", color="tab:blue", linewidth=7.0, y_offset=-0.10, alpha=0.90, zorder=2)
    plot_intervals_as_steps(ax2, pred_intervals, LABEL_TO_INT, label_name="SVM", color="tab:orange", linewidth=3.0, y_offset=0.10, alpha=0.95, zorder=3)

    ax2.set_yticks(range(len(LABEL_ORDER)))
    ax2.set_yticklabels(LABEL_ORDER)
    ax2.set_ylim(-0.5, len(LABEL_ORDER) - 0.5)
    ax2.set_xlabel("Czas [ms]")
    ax2.set_ylabel("Manewr")
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def save_combined_crop_figure(raw_dfs, pred_dfs, out_path):
    fig, axes = plt.subplots(2, len(CROP_ORDER), figsize=(18, 5.5), dpi=100, gridspec_kw={"height_ratios": [3.0, 1.35]}, sharey=False)

    for col_idx, rec_name in enumerate(CROP_ORDER):
        raw_df = raw_dfs[rec_name]
        pred_df = pred_dfs[rec_name]
        segments = DATASETS[rec_name]["segments"]

        t0, t1 = CROP_INTERVALS[rec_name]
        t_max = float(raw_df["time"].max())

        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]

        plot_rpm_channels(ax=ax_top, raw_df=raw_df, t0=t0, t1=t1, y_min=13000, y_max=20000, show_legend=(col_idx == 0))

        ax_top.set_title(CROP_TITLES[rec_name], fontsize=12)
        ax_top.set_xlim(t0, t1)
        ax_top.set_ylabel("RPM" if col_idx == 0 else "")

        gt_intervals = gt_segments_to_intervals(segments, t_max=t_max, t0=t0, t1=t1)
        pred_part = pred_df[(pred_df["window_end_ms"] >= t0) & (pred_df["window_start_ms"] <= t1)].copy()
        pred_intervals = prediction_rows_to_intervals(pred_part, label_col="pred_label", t0=t0, t1=t1)

        plot_intervals_as_steps(ax_bottom, gt_intervals, LABEL_TO_INT, label_name="GT", color="tab:blue", linewidth=7.0, y_offset=-0.10, alpha=0.90, zorder=2)
        plot_intervals_as_steps(ax_bottom, pred_intervals, LABEL_TO_INT, label_name="SVM", color="tab:orange", linewidth=3.0, y_offset=0.10, alpha=0.95, zorder=3)

        ax_bottom.set_xlim(t0, t1)
        ax_bottom.set_yticks(range(len(LABEL_ORDER)))
        ax_bottom.set_yticklabels(LABEL_ORDER, fontsize=8)
        ax_bottom.set_ylim(-0.5, len(LABEL_ORDER) - 0.5)
        ax_bottom.grid(True, alpha=0.35)
        ax_bottom.set_xlabel("Czas [ms]")
        ax_bottom.set_ylabel("Manewr" if col_idx == 0 else "")

        if col_idx == 0:
            ax_bottom.legend(fontsize=8, loc="upper right")

    fig.suptitle("Wybrane fragmenty nagrań: RPM oraz ground truth vs predykcja SVM", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def evaluate_model(model, X_eval, y_eval, prefix, out_dir, report_lines, save_plots=True):
    y_pred = model.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred)
    bal_acc = balanced_accuracy_score(y_eval, y_pred)
    macro_f1 = f1_score(y_eval, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_eval, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)

    report_dict = classification_report(y_eval, y_pred, labels=LABEL_ORDER, digits=4, zero_division=0, output_dict=True)
    text_report = classification_report(y_eval, y_pred, labels=LABEL_ORDER, digits=4, zero_division=0)

    print(f"\n=== {prefix} ===")
    print("Accuracy:", acc)
    print("Balanced accuracy:", bal_acc)
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)
    print(text_report)

    report_lines.append(f"\n=== {prefix} ===\n")
    report_lines.append(f"Accuracy: {acc:.4f}\n")
    report_lines.append(f"Balanced accuracy: {bal_acc:.4f}\n")
    report_lines.append(f"Macro F1: {macro_f1:.4f}\n")
    report_lines.append(f"Weighted F1: {weighted_f1:.4f}\n\n")
    report_lines.append("Support:\n")
    report_lines.append(str(pd.Series(y_eval).value_counts()) + "\n\n")
    report_lines.append(text_report + "\n")

    cm = confusion_matrix(y_eval, y_pred, labels=LABEL_ORDER)
    cm_norm = confusion_matrix(y_eval, y_pred, labels=LABEL_ORDER, normalize="true")

    pd.DataFrame(cm, index=[f"true_{x}" for x in LABEL_ORDER], columns=[f"pred_{x}" for x in LABEL_ORDER]).to_csv(out_dir / f"{prefix}_confusion_matrix_raw.csv")
    pd.DataFrame(cm_norm, index=[f"true_{x}" for x in LABEL_ORDER], columns=[f"pred_{x}" for x in LABEL_ORDER]).to_csv(out_dir / f"{prefix}_confusion_matrix_normalized.csv")

    if save_plots:
        save_confusion_matrix_plot(cm, LABEL_ORDER, out_dir / f"{prefix}_confusion_matrix_raw.png", title=f"{prefix}: macierz pomyłek, acc={acc:.3f}", values_format="d")
        save_confusion_matrix_plot(cm_norm, LABEL_ORDER, out_dir / f"{prefix}_confusion_matrix_normalized.png", title=f"{prefix}: macierz znormalizowana, bal acc={bal_acc:.3f}", values_format=".2f")

    per_class = {}
    for label in LABEL_ORDER:
        per_class[label] = {
            "precision": float(report_dict.get(label, {}).get("precision", 0.0)),
            "recall": float(report_dict.get(label, {}).get("recall", 0.0)),
            "f1": float(report_dict.get(label, {}).get("f1-score", 0.0)),
            "support": int(report_dict.get(label, {}).get("support", 0)),
        }

    min_precision = min(per_class[x]["precision"] for x in LABEL_ORDER)
    min_recall = min(per_class[x]["recall"] for x in LABEL_ORDER)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "min_precision": float(min_precision),
        "min_recall": float(min_recall),
        "per_class": per_class,
    }


def run_experiment(window_ms=WINDOW_MS, step_ms=STEP_MS, test_size=TEST_SIZE, out_dir=OUT_DIR_DEFAULT, make_plots=True, save_models=True, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dfs = {}
    all_windows = []

    for recording_name, spec in DATASETS.items():
        raw_df = load_rpm_csv(spec["file"])
        raw_dfs[recording_name] = raw_df

        windows = build_windows_for_dataset(
            recording_name=recording_name,
            raw_df=raw_df,
            segments=spec["segments"],
            window_ms=window_ms,
            step_ms=step_ms,
            min_samples=MIN_SAMPLES_IN_WINDOW,
            test_size=test_size,
        )

        if len(windows) == 0:
            if verbose:
                print(f"[WARN] Brak okien dla {recording_name}")
            continue

        windows["source_file"] = str(spec["file"])
        all_windows.append(windows)

        if verbose:
            print(f"{recording_name:14s} | okna: {len(windows)}")
            print(windows["label"].value_counts())
            print(windows["split"].value_counts())
            print()

    if not all_windows:
        raise RuntimeError("Nie utworzono żadnych okien.")

    data = pd.concat(all_windows, ignore_index=True)

    meta_cols = {
        "recording", "label", "segment_id", "segment_start_ms", "segment_end_ms",
        "window_start_ms", "window_end_ms", "window_center_ms", "split", "source_file",
    }

    feature_cols = [c for c in data.columns if c not in meta_cols]

    train_data = data[data["split"] == "train"].copy()
    test_data = data[data["split"] == "test"].copy()
    ignored_data = data[data["split"] == "ignore_boundary"].copy()

    if len(train_data) == 0 or len(test_data) == 0:
        raise RuntimeError("Brak danych train albo test. Sprawdź TEST_SIZE, WINDOW_MS i GT.")

    train_data_balanced = downsample_hover_train(train_data, hover_ratio_to_max_other=HOVER_RATIO_TO_MAX_OTHER, random_state=RANDOM_STATE)

    X_train = train_data_balanced[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_data_balanced["label"].astype(str).reset_index(drop=True)

    train_target_per_class = int(y_train.value_counts().max())

    if AUGMENT_TRAIN:
        X_train_aug, y_train_aug = augment_features_to_target(
            X_train, y_train,
            target_per_class=train_target_per_class,
            classes=LABEL_ORDER,
            noise_frac=AUGMENT_TRAIN_NOISE_FRAC,
            random_state=RANDOM_STATE,
            allow_downsample=False,
        )
    else:
        X_train_aug, y_train_aug = X_train, y_train

    X_test_raw = test_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test_raw = test_data["label"].astype(str).reset_index(drop=True)

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    )

    model.fit(X_train_aug, y_train_aug)

    report = []
    report.append("Klasyfikacja manewrów drona — SVM\n")
    report.append("=" * 60 + "\n\n")
    report.append(f"WINDOW_MS = {window_ms}\n")
    report.append(f"STEP_MS = {step_ms}\n")
    report.append(f"MIN_SAMPLES_IN_WINDOW = {MIN_SAMPLES_IN_WINDOW}\n")
    report.append(f"TEST_SIZE = {test_size}\n\n")
    report.append("Split: czasowy wewnątrz każdego segmentu GT, bez nakładania train/test.\n\n")
    report.append(f"HOVER_RATIO_TO_MAX_OTHER = {HOVER_RATIO_TO_MAX_OTHER}\n")
    report.append(f"AUGMENT_TRAIN = {AUGMENT_TRAIN}\n")
    report.append(f"TRAIN_TARGET_PER_CLASS = {train_target_per_class}\n")
    report.append(f"AUGMENT_TRAIN_NOISE_FRAC = {AUGMENT_TRAIN_NOISE_FRAC}\n")
    report.append(f"AUGMENT_TEST_FOR_METRICS = {AUGMENT_TEST_FOR_METRICS}\n")
    report.append(f"BALANCED_TEST_TARGET_PER_CLASS = {BALANCED_TEST_TARGET_PER_CLASS}\n")
    report.append(f"AUGMENT_TEST_NOISE_FRAC = {AUGMENT_TEST_NOISE_FRAC}\n\n")
    report.append("Liczba okien per klasa — całość:\n")
    report.append(str(data["label"].value_counts()) + "\n\n")
    report.append("Liczba okien per klasa — train przed balansem:\n")
    report.append(str(train_data["label"].value_counts()) + "\n\n")
    report.append("Liczba okien per klasa — train po downsamplingu:\n")
    report.append(str(train_data_balanced["label"].value_counts()) + "\n\n")
    report.append("Liczba próbek treningowych po augmentacji do targetu:\n")
    report.append(str(y_train_aug.value_counts()) + "\n\n")
    report.append("Liczba okien per klasa — test raw:\n")
    report.append(str(test_data["label"].value_counts()) + "\n\n")
    report.append("Okna pominięte przez granicę train/test:\n")
    report.append(str(ignored_data["label"].value_counts()) + "\n\n")

    if verbose:
        print("\nLiczba okien per klasa — całość:")
        print(data["label"].value_counts())
        print("\nLiczba okien per klasa — train przed balansem:")
        print(train_data["label"].value_counts())
        print("\nLiczba okien per klasa — train po downsamplingu hover:")
        print(train_data_balanced["label"].value_counts())
        print("\nLiczba próbek treningowych po augmentacji do targetu:")
        print(y_train_aug.value_counts())
        print("\nLiczba okien per klasa — test raw:")
        print(test_data["label"].value_counts())
        print("\nOkna pominięte przez granicę train/test:")
        print(ignored_data["label"].value_counts() if len(ignored_data) else "brak")

    raw_metrics = evaluate_model(model, X_test_raw, y_test_raw, "raw_test", out_dir, report, save_plots=make_plots)

    if AUGMENT_TEST_FOR_METRICS:
        X_test_bal, y_test_bal = augment_features_to_target(
            X_test_raw, y_test_raw,
            target_per_class=BALANCED_TEST_TARGET_PER_CLASS,
            classes=LABEL_ORDER,
            noise_frac=AUGMENT_TEST_NOISE_FRAC,
            random_state=RANDOM_STATE + 999,
            allow_downsample=True,
        )

        report.append("\nLiczba próbek balanced augmented test:\n")
        report.append(str(y_test_bal.value_counts()) + "\n\n")

        if verbose:
            print("\nLiczba próbek balanced augmented test:")
            print(y_test_bal.value_counts())

        balanced_augmented_metrics = evaluate_model(model, X_test_bal, y_test_bal, "balanced_augmented_test", out_dir, report, save_plots=make_plots)
    else:
        balanced_augmented_metrics = None

    with open(out_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("".join(report))

    data_with_pred = data.copy()
    X_all = data_with_pred[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data_with_pred["pred_label"] = model.predict(X_all)

    proba = model.predict_proba(X_all)
    for i, cls in enumerate(model.classes_):
        data_with_pred[f"p_{cls}"] = proba[:, i]
    data_with_pred["confidence"] = np.max(proba, axis=1)
    data_with_pred.to_csv(out_dir / "all_windows_features_with_predictions.csv", index=False)

    pred_dfs = {}
    if make_plots:
        for recording_name, spec in DATASETS.items():
            pred_df = data_with_pred[data_with_pred["recording"] == recording_name].copy()
            pred_df = pred_df.sort_values("window_center_ms").reset_index(drop=True)
            pred_dfs[recording_name] = pred_df
            pred_df.to_csv(out_dir / f"predictions_{recording_name}.csv", index=False)

            plot_recording_prediction(recording_name, raw_dfs[recording_name], pred_df, spec["segments"], out_dir / f"plot_{recording_name}.png")

        save_combined_crop_figure(raw_dfs, pred_dfs, out_dir / "summary_cropped_fragments.png")

    if save_models:
        eval_bundle = {
            "model": model,
            "feature_cols": feature_cols,
            "label_order": LABEL_ORDER,
            "window_ms": window_ms,
            "step_ms": step_ms,
            "trained_on": "temporal_train_split_with_balanced_train_augmentation",
            "raw_test_metrics": raw_metrics,
            "balanced_augmented_test_metrics": balanced_augmented_metrics,
        }
        joblib.dump(eval_bundle, out_dir / "svm_maneuver_model_eval.joblib")

        final_source = data[data["split"].isin(["train", "test"])].copy()
        final_source_balanced = downsample_hover_train(final_source, hover_ratio_to_max_other=HOVER_RATIO_TO_MAX_OTHER, random_state=RANDOM_STATE)
        X_final = final_source_balanced[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_final = final_source_balanced["label"].astype(str).reset_index(drop=True)
        final_target = int(y_final.value_counts().max())
        X_final_aug, y_final_aug = augment_features_to_target(
            X_final, y_final,
            target_per_class=final_target,
            classes=LABEL_ORDER,
            noise_frac=AUGMENT_TRAIN_NOISE_FRAC,
            random_state=RANDOM_STATE + 123,
            allow_downsample=False,
        )
        final_model = make_pipeline(
            StandardScaler(),
            SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        )
        final_model.fit(X_final_aug, y_final_aug)
        final_bundle = {
            "model": final_model,
            "feature_cols": feature_cols,
            "label_order": LABEL_ORDER,
            "window_ms": window_ms,
            "step_ms": step_ms,
            "trained_on": "all_non_boundary_data_with_balanced_augmentation",
        }
        joblib.dump(final_bundle, out_dir / "svm_maneuver_model.joblib")

    result = {
        "status": "ok",
        "window_ms": float(window_ms),
        "step_ms": float(step_ms),
        "test_size": float(test_size),
        "n_windows_all": int(len(data)),
        "n_train_raw": int(len(train_data)),
        "n_train_balanced": int(len(train_data_balanced)),
        "n_train_augmented": int(len(y_train_aug)),
        "n_test_raw": int(len(test_data)),
        "n_ignore_boundary": int(len(ignored_data)),
    }

    for prefix, metrics in [("raw_test", raw_metrics), ("balanced_augmented_test", balanced_augmented_metrics)]:
        if metrics is None:
            continue
        result[f"{prefix}_accuracy"] = metrics["accuracy"]
        result[f"{prefix}_balanced_accuracy"] = metrics["balanced_accuracy"]
        result[f"{prefix}_macro_f1"] = metrics["macro_f1"]
        result[f"{prefix}_weighted_f1"] = metrics["weighted_f1"]
        result[f"{prefix}_min_precision"] = metrics["min_precision"]
        result[f"{prefix}_min_recall"] = metrics["min_recall"]
        for label in LABEL_ORDER:
            result[f"{prefix}_{label}_precision"] = metrics["per_class"][label]["precision"]
            result[f"{prefix}_{label}_recall"] = metrics["per_class"][label]["recall"]
            result[f"{prefix}_{label}_f1"] = metrics["per_class"][label]["f1"]
            result[f"{prefix}_{label}_support"] = metrics["per_class"][label]["support"]

    if verbose:
        print(f"\nZapisano wyniki do: {out_dir}")
    return result


def main():
    run_experiment(window_ms=WINDOW_MS, step_ms=STEP_MS, test_size=TEST_SIZE, out_dir=OUT_DIR_DEFAULT, make_plots=True, save_models=True, verbose=True)


if __name__ == "__main__":
    main()
