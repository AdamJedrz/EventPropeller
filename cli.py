import argparse

from config import PipelineConfig
from pipeline import run_rpm_pipeline


def main():
    parser = argparse.ArgumentParser(description="Estimate propeller RPM from event-camera .dat recording.")
    parser.add_argument("dat_path")
    parser.add_argument("--out", default="rpm.csv", help="Output CSV path")
    parser.add_argument("--n-propellers", type=int, default=2)
    parser.add_argument("--bundle-us", type=int, default=2000)
    parser.add_argument("--bundles-per-window", type=int, default=10)
    parser.add_argument("--polarity", choices=["positive", "negative", "all"], default="positive")
    parser.add_argument("--preview", action="store_true", help="Show OpenCV preview windows")
    parser.add_argument("--preview-wait-ms", type=int, default=1)
    args = parser.parse_args()

    config = PipelineConfig(
        n_propellers=args.n_propellers,
        bundle_us=args.bundle_us,
        bundles_per_window=args.bundles_per_window,
        polarity_mode=args.polarity,
        preview=args.preview,
        preview_wait_ms=args.preview_wait_ms,
    )
    rpm_df = run_rpm_pipeline(args.dat_path, config)
    rpm_df.to_csv(args.out)
    print(rpm_df)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
