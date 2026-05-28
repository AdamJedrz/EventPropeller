from pathlib import Path

from config import PipelineConfig
from dat import count_events, get_recording_duration
from pipeline import run_rpm_pipeline
import numpy as np
import pandas as pd


EVENTS_DIR = Path("events")
DAT_FILE = EVENTS_DIR / "climb_ir.dat"
OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "rpm_results_climb_prooba.csv"


config = PipelineConfig(
    # ============================================================
    # Dane / podział czasu
    # ============================================================
    n_propellers=4,                  # Ile śmigieł/tracków ma być śledzonych.
    chunk_size=5_000_000,            # Ile eventów czytać naraz z pliku .dat.
    bundle_us=1000,                  # Długość bundla do RPM [us]. 2000 us = 2 ms.
    bundles_per_window=20,           # Ile bundli w oknie ROI. 10 * 2 ms = 20 ms.
    polarity_mode="negative",        # "positive", "negative" albo "all".

    # ============================================================
    # Start od środka pliku / limit do diagnostyki
    # ============================================================
    start_time_ms=0,                 # 0 = od początku. Np. 40_000 = zacznij od 40 s.
    max_windows=None,                # None = cały plik. Np. 5 = tylko 5 okien i stop.

    # ============================================================
    # Filtracja gęstościowa tła
    # ============================================================
    bin_size=6,                      # Rozmiar binu przestrzennego [px]. Większy = bardziej zgrubna filtracja.
    min_events_in_bin=20,             # Minimalna liczba eventów w binie.
    density_min_component_area=4,   # Minimalny obszar składowej na masce binów.
    remove_border_components=False,   # Usuwanie komponentów dotykających brzegu obrazu.

    # ============================================================
    # Opcjonalna morfologia na mapie aktywnych binów
    # ============================================================
    density_bin_morph_mode="erode",       # "none", "erode" albo "open". Działa na binach, nie na pikselowej masce.
    density_bin_morph_kernel_size=3,      # Kernel na mapie binów.
    density_bin_morph_iterations=1,       # 0 = wyłączone.

    # ============================================================
    # Klasteryzacja eventów na 4 śmigła
    # ============================================================
    cluster_bbox_pad=2,              # Margines bbox wokół klastra. Większy = więcej eventów do MC.
    cluster_min_events_per_cluster=40, # Minimalna liczba eventów w klastrze śmigła.
    cluster_sample_max_events=10000, # Limit punktów do dopasowania klastrów dla szybkości.
    cluster_attempts=5,              # Liczba prób inicjalizacji klasteryzacji.

    # ============================================================
    # Tracking centroidów między oknami
    # ============================================================
    bootstrap_sort="x",              # Kolejność początkowych tracków: "x" albo "y".
    max_centroid_dx_per_window=50.0, # Maks. przesunięcie centroidu w X na okno.
    max_centroid_dy_per_window=50.0, # Maks. przesunięcie centroidu w Y na okno.
    max_centroid_distance_per_window=None, # Opcjonalny limit dystansu euklidesowego.

    # ============================================================
    # Przeszukiwanie RPM
    # ============================================================
    rpm_abs_max=20000,                # Globalny zakres RPM: -8000..8000.
    rpm_step_coarse=200,             # Krok globalnego przeszukiwania. Większy = szybciej, mniej dokładnie.
    rpm_step_fine=20,                # Krok doprecyzowania po globalnym searchu.
    rpm_refine_span=200,             # Zakres doprecyzowania wokół najlepszego coarse RPM.
    rpm_local_delta=300,             # Lokalne szukanie: prev_rpm +/- ta wartość.
    full_search_period_us=20_000_000,   # Co ile robić pełny search zamiast lokalnego [us].
    min_events_for_rpm=200,           # Minimalna liczba eventów w ROI do liczenia RPM.

    # ============================================================
    # Downsampling w motion compensation
    # ============================================================
    max_events_mc=5000,              # Maks. eventów na bundle/śmigło. Mniej = szybciej, możliwie mniej stabilnie.
    downsample_time_bins=20,         # Liczba koszyków czasowych przy downsamplingu.

    # ============================================================
    # Backend motion compensation
    # ============================================================
    parallel_mc=True,               # False = sekwencyjnie NumPy CPU. True = równolegle na GPU przez PyTorch/CUDA, jeśli dostępne.
    parallel_candidate_chunk_size=1024, # Ile par RPM-środek liczyć naraz na GPU. Zmniejsz, jeśli brakuje VRAM.

    # ============================================================
    # Funkcja celu
    # ============================================================
    score_mode="eventpro",        # Dostępne: "mean_square", "variance", "inverse_occupancy", "eventpro".
    mc_score_lambda=1.0,             # Używane tylko dla score_mode="eventpro".
    mc_score_eps=1e-6,               # Używane dla score_mode="eventpro" oraz "inverse_occupancy".

    # ============================================================
    # Czas referencyjny warpingu w bundlu
    # ============================================================
    reference_time_fractions=(0.5,), # Jedna wartość = jeden warping. 0.5 = środek bundla.

    # ============================================================
    # Opcjonalna optymalizacja środka obrotu
    # ============================================================
    center_search_radius_px=20.0,     # 0 = wyłączone. Np. 5 szuka środka w promieniu 5 px.
    center_search_step_px=4.0,       # Krok siatki środka. Duży promień i mały krok bardzo spowalnia.

    # ============================================================
    # Opcjonalny eliptyczny warping q=b/a
    # ============================================================
    q_search_enabled=False,           # False = stary model kołowy/fixed q. True = optymalizuj q po siatce.
    q_fixed=1.0,                      # Używane, gdy q_search_enabled=False. 1.0 = pełny okrąg, czyli stary model.
    q_min=0.90,                       # Minimalne q=b/a w searchu. 0.50 odpowiada mocnemu spłaszczeniu elipsy.
    q_max=1.00,                       # Maksymalne q. 1.00 zawsze zawiera model kołowy jako kandydat.
    q_step=0.02,                      # Krok q. 0.05 daje 11 kandydatów dla zakresu 0.50..1.00.
    q_axis_angle_deg=0.0,             # Orientacja dużej osi elipsy w obrazie. 0 = poziomo, 90 = pionowo.
    q_axis_search_enabled=False,       # True = optymalizuj orientację dużej osi elipsy po siatce.
    q_axis_angle_min_deg=0.0,         # Minimalny kąt osi elipsy [deg].
    q_axis_angle_max_deg=160.0,       # Maksymalny kąt osi elipsy [deg]. 180 == 0, więc zwykle max 165 przy kroku 15.
    q_axis_angle_step_deg=20.0,       # Krok kąta osi. Mniejszy = dokładniej, ale wolniej.

    # ============================================================
    # Podgląd / logowanie
    # ============================================================
    preview=True,                   # True = pokaż okna OpenCV.
    preview_wait_ms=5000,            # Czas oczekiwania na preview. ESC przerywa.
    preview_every_n_windows=10,       # Pokaż preview co N okien.
    verbose_chunks=True,             # Wypisuj postęp czytania chunków.
)


def get_dat_path() -> Path:
    if not DAT_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {DAT_FILE.resolve()}")
    return DAT_FILE


def print_recording_info(dat_path: Path):
    print("=" * 80)
    print("INFORMACJE O NAGRANIU")
    print("=" * 80)

    print(f"event_count: {count_events(str(dat_path))}")
    for key, value in get_recording_duration(str(dat_path)).items():
        print(f"{key}: {value}")

    print("=" * 80)
    print()


def _find_column_by_prefix(columns, prefix):
    for col in columns:
        if str(col).startswith(prefix):
            return col
    return None


def make_export_dataframe(rpm_df, config: PipelineConfig):
    export_df = rpm_df.copy()
    out = export_df.copy()

    result = {
        "time": export_df.index.to_numpy(dtype=float) * config.bundle_us / 1000.0,
    }

    for tid in range(config.n_propellers):
        rpm_col = _find_column_by_prefix(export_df.columns, f"C{tid}")
        q_col = _find_column_by_prefix(export_df.columns, f"Q{tid}")
        axis_col = _find_column_by_prefix(export_df.columns, f"A{tid}")

        result[f"c{tid + 1}"] = out[rpm_col].to_numpy() if rpm_col is not None else np.full(len(out), np.nan)
        result[f"q{tid + 1}"] = out[q_col].to_numpy() if q_col is not None else np.full(len(out), np.nan)
        result[f"axis{tid + 1}_deg"] = out[axis_col].to_numpy() if axis_col is not None else np.full(len(out), np.nan)

    export_df = pd.DataFrame(result)

    window_ms = config.bundle_us * config.bundles_per_window / 1000.0
    if float(window_ms).is_integer():
        export_df["time"] = export_df["time"].astype(int)

    return export_df


def main():
    dat_path = get_dat_path()

    print("=" * 80)
    print("PROP RPM PIPELINE")
    print("=" * 80)
    print(f"Plik wejściowy: {dat_path.resolve()}")
    print(f"Preview: {config.preview}")
    print(f"Okno: {config.bundle_us * config.bundles_per_window / 1000:.1f} ms")
    print(f"Start time: {config.start_time_ms} ms")
    print(f"Max windows: {config.max_windows}")
    print(f"Score mode: {config.score_mode}")
    print(f"q search: {config.q_search_enabled} | q={config.q_min}..{config.q_max} step {config.q_step}")
    print(f"axis search: {config.q_axis_search_enabled} | axis={config.q_axis_angle_min_deg}..{config.q_axis_angle_max_deg} step {config.q_axis_angle_step_deg} deg")
    print(f"Parallel: {config.parallel_mc}")
    print(f"Torch candidate chunk size: {config.parallel_candidate_chunk_size}")
    print(f"Reference fractions: {config.reference_time_fractions}")
    print(f"Max events MC: {config.max_events_mc}")
    print(f"Density bin morph: {config.density_bin_morph_mode} | kernel={config.density_bin_morph_kernel_size} | iter={config.density_bin_morph_iterations}")
    print()

    print_recording_info(dat_path)
    OUTPUT_DIR.mkdir(exist_ok=True)

    rpm_df = run_rpm_pipeline(str(dat_path), config)
    export_df = make_export_dataframe(rpm_df, config)
    export_df.to_csv(OUTPUT_CSV, index=False)

    print()
    print("=" * 80)
    print("PRZERWANO" if rpm_df.attrs.get("interrupted", False) else "GOTOWE")
    print("=" * 80)
    print(f"Wyniki zapisane do: {OUTPUT_CSV.resolve()}")
    print()
    print(export_df.head())


if __name__ == "__main__":
    main()