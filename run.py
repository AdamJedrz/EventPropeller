from pathlib import Path

from config import PipelineConfig
from dat import count_events, get_recording_duration
from pipeline import run_rpm_pipeline


EVENTS_DIR = Path("events")
DAT_FILE = EVENTS_DIR / "30.dat"
OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "rpm_results.csv"


config = PipelineConfig(
    # ============================================================
    # Dane / podział czasu
    # ============================================================
    n_propellers=2,                  # Ile śmigieł/tracków ma być śledzonych.
    chunk_size=5_000_000,            # Ile eventów czytać naraz z pliku .dat.
    bundle_us=2000,                  # Długość bundla do RPM [us]. 2000 us = 2 ms.
    bundles_per_window=10,           # Ile bundli w oknie ROI. 10 * 2 ms = 20 ms.
    polarity_mode="positive",        # "positive", "negative" albo "all".

    # ============================================================
    # Start od środka pliku / limit do diagnostyki
    # ============================================================
    start_time_ms=0,                 # 0 = od początku. Np. 40_000 = zacznij od 40 s.
    max_windows=None,                # None = cały plik. Np. 5 = tylko 5 okien i stop.

    # ============================================================
    # Filtracja gęstościowa tła
    # ============================================================
    bin_size=7,                      # Rozmiar binu przestrzennego [px]. Większy = bardziej zgrubna filtracja.
    min_events_in_bin=3,             # Minimalna liczba eventów w binie.
    density_min_component_area=18,   # Minimalny obszar składowej na masce binów.
    remove_border_components=True,   # Usuwanie komponentów dotykających brzegu obrazu.

    # ============================================================
    # Maska ROI wirników
    # binary -> dilate -> close -> remove small -> fill holes
    # ============================================================
    mask_dilate_kernel_size=5,       # Kernel dylatacji. Większy = mocniej skleja eventy.
    mask_dilate_iterations=1,        # Liczba iteracji dylatacji.
    mask_close_kernel_size=9,        # Kernel domknięcia. Większy = mocniej domyka dziury/przerwy.
    mask_close_iterations=1,         # Liczba iteracji domknięcia.
    mask_fill_holes=True,            # Wypełnianie dziur w masce.
    mask_min_component_area_px=3000, # Minimalny obszar ROI śmigła [px].
    max_detection_components=10,     # Maks. liczba kandydatów z segmentacji.

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
    rpm_abs_max=8000,                # Globalny zakres RPM: -8000..8000.
    rpm_step_coarse=200,             # Krok globalnego przeszukiwania. Większy = szybciej, mniej dokładnie.
    rpm_step_fine=20,                # Krok doprecyzowania po globalnym searchu.
    rpm_refine_span=200,             # Zakres doprecyzowania wokół najlepszego coarse RPM.
    rpm_local_delta=150,             # Lokalne szukanie: prev_rpm +/- ta wartość.
    full_search_period_us=500_000,   # Co ile robić pełny search zamiast lokalnego [us].
    min_events_for_rpm=50,           # Minimalna liczba eventów w ROI do liczenia RPM.

    # ============================================================
    # Downsampling w motion compensation
    # ============================================================
    max_events_mc=3000,              # Maks. eventów na bundle/śmigło. Mniej = szybciej, możliwie mniej stabilnie.
    downsample_time_bins=20,         # Liczba koszyków czasowych przy downsamplingu.

    # ============================================================
    # Funkcja celu
    # ============================================================
    score_mode="mean_square",        # "mean_square" = szybka, score=mean(H^2). "eventpro" = wolniejsza z exp().
    mc_score_lambda=1.0,             # Używane tylko dla score_mode="eventpro".
    mc_score_eps=1e-6,               # Używane tylko dla score_mode="eventpro".

    # ============================================================
    # Czas referencyjny warpingu w bundlu
    # ============================================================
    reference_time_fractions=(0.5,), # Jedna wartość = jeden warping. 0.5 = środek bundla.

    # ============================================================
    # Opcjonalna optymalizacja środka obrotu
    # ============================================================
    center_search_radius_px=0,     # 0 = wyłączone. Np. 5 szuka środka w promieniu 5 px.
    center_search_step_px=20.0,       # Krok siatki środka. Duży promień i mały krok bardzo spowalnia.

    # ============================================================
    # Podgląd / logowanie
    # ============================================================
    preview=False,                   # True = pokaż okna OpenCV.
    preview_wait_ms=5000,            # Czas oczekiwania na preview. ESC przerywa.
    preview_every_n_windows=1,       # Pokaż preview co N okien.
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


def make_export_dataframe(rpm_df, config: PipelineConfig):
    export_df = rpm_df.copy()
    export_df.insert(0, "time", export_df.index.to_numpy(dtype=float) * config.bundle_us / 1000.0)

    rename_map = {
        old_name: f"c{i + 1}"
        for i, old_name in enumerate(export_df.columns[1:])
    }
    export_df = export_df.rename(columns=rename_map)

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
    print(f"Reference fractions: {config.reference_time_fractions}")
    print(f"Max events MC: {config.max_events_mc}")
    print()

    print_recording_info(dat_path)
    OUTPUT_DIR.mkdir(exist_ok=True)

    rpm_df = run_rpm_pipeline(str(dat_path), config)
    export_df = make_export_dataframe(rpm_df, config)
    export_df.to_csv(OUTPUT_CSV, index=False)

    print()
    print("=" * 80)
    print("GOTOWE")
    print("=" * 80)
    print(f"Wyniki zapisane do: {OUTPUT_CSV.resolve()}")
    print()
    print(export_df.head())


if __name__ == "__main__":
    main()
