"""
Batch process all cluster load profiles to 15-minute week-averaged format.
"""

import pandas as pd


def process_cori():
    """Process Cori: 30-second sampling, peak 5.7 MW."""
    print("\n" + "=" * 60)
    print("Processing Cori (30s sampling, 5.7 MW peak)")
    print("=" * 60)

    df = pd.read_csv("Cori_power_30_sec.csv")
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Week average
    seconds_per_day = 86400
    seconds_per_week = seconds_per_day * 7
    df["time_of_week"] = df["timestamp_secs"] % seconds_per_week
    weekly_profile = df.groupby("time_of_week")["measured_kW"].mean().reset_index()
    weekly_profile.columns = ["timestamp_s", "kW"]

    # Normalize
    peak_mw = 5.7
    weekly_profile["MW"] = weekly_profile["kW"] / 1000
    weekly_profile["watts"] = weekly_profile["MW"] / peak_mw

    # Resample to 15 min (7 days * 96 intervals = 672 points)
    max_time = seconds_per_week
    new_timestamps = list(range(0, max_time, 900))
    result = pd.DataFrame(
        {
            "timestamp_s": new_timestamps,
            "timestamp_hr": [t / 3600 for t in new_timestamps],
            "day_of_week": [t // seconds_per_day for t in new_timestamps],
        }
    )

    # Interpolate
    result["watts"] = (
        weekly_profile.set_index("timestamp_s")["watts"]
        .reindex(new_timestamps, method=None)
        .interpolate(method="linear")
        .values
    )

    # Save
    result.to_csv("cori_week_avg.csv", index=False)

    actual_peak = weekly_profile["MW"].max()
    actual_avg = weekly_profile["MW"].mean()
    print(f"Peak: {actual_peak:.2f} MW ({actual_peak / peak_mw * 100:.1f}% of {peak_mw} MW)")
    print(f"Average: {actual_avg:.2f} MW ({actual_avg / peak_mw * 100:.1f}% of {peak_mw} MW)")
    print(
        f"Output: {len(result)} points (7 days × 96 intervals), range {result['watts'].min():.3f} to {result['watts'].max():.3f}"
    )


def process_hawk():
    """Process Hawk: 15-minute sampling already, peak 3.45 MW."""
    print("\n" + "=" * 60)
    print("Processing Hawk (15min sampling, 3.45 MW peak)")
    print("=" * 60)

    df = pd.read_csv("Hawk_power_15_min.csv")
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Week average
    seconds_per_day = 86400
    seconds_per_week = seconds_per_day * 7
    df["time_of_week"] = df["timestamp_secs"] % seconds_per_week
    weekly_profile = df.groupby("time_of_week")["measured_kW"].mean().reset_index()
    weekly_profile.columns = ["timestamp_s", "kW"]

    # Normalize
    peak_mw = 3.45
    weekly_profile["MW"] = weekly_profile["kW"] / 1000
    weekly_profile["watts"] = weekly_profile["MW"] / peak_mw
    weekly_profile["timestamp_hr"] = weekly_profile["timestamp_s"] / 3600
    weekly_profile["day_of_week"] = weekly_profile["timestamp_s"] // seconds_per_day

    # Already 15 min, just select columns
    result = weekly_profile[["timestamp_s", "watts", "timestamp_hr", "day_of_week"]].copy()

    # Save
    result.to_csv("hawk_week_avg.csv", index=False)

    actual_peak = weekly_profile["MW"].max()
    actual_avg = weekly_profile["MW"].mean()
    print(f"Peak: {actual_peak:.2f} MW ({actual_peak / peak_mw * 100:.1f}% of {peak_mw} MW)")
    print(f"Average: {actual_avg:.2f} MW ({actual_avg / peak_mw * 100:.1f}% of {peak_mw} MW)")
    print(
        f"Output: {len(result)} points (7 days × 96 intervals), range {result['watts'].min():.3f} to {result['watts'].max():.3f}"
    )


def process_lumi():
    """Process Lumi: 10-minute sampling, peak 7.973 MW."""
    print("\n" + "=" * 60)
    print("Processing Lumi (10min sampling, 7.973 MW peak)")
    print("=" * 60)

    df = pd.read_csv("Lumi_power_10_min.csv")
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Week average
    seconds_per_day = 86400
    seconds_per_week = seconds_per_day * 7
    df["time_of_week"] = df["timestamp_secs"] % seconds_per_week
    weekly_profile = df.groupby("time_of_week")["measured_kW"].mean().reset_index()
    weekly_profile.columns = ["timestamp_s", "kW"]

    # Normalize
    peak_mw = 7.973
    weekly_profile["MW"] = weekly_profile["kW"] / 1000
    weekly_profile["watts"] = weekly_profile["MW"] / peak_mw

    # Resample from 10 min to 15 min across the week
    max_time = seconds_per_week
    new_timestamps = list(range(0, max_time, 900))
    result = pd.DataFrame(
        {
            "timestamp_s": new_timestamps,
            "timestamp_hr": [t / 3600 for t in new_timestamps],
            "day_of_week": [t // seconds_per_day for t in new_timestamps],
        }
    )

    # Interpolate
    result["watts"] = (
        weekly_profile.set_index("timestamp_s")["watts"]
        .reindex(new_timestamps, method=None)
        .interpolate(method="linear")
        .values
    )

    # Save
    result.to_csv("lumi_week_avg.csv", index=False)

    actual_peak = weekly_profile["MW"].max()
    actual_avg = weekly_profile["MW"].mean()
    print(f"Peak: {actual_peak:.2f} MW ({actual_peak / peak_mw * 100:.1f}% of {peak_mw} MW)")
    print(f"Average: {actual_avg:.2f} MW ({actual_avg / peak_mw * 100:.1f}% of {peak_mw} MW)")
    print(
        f"Output: {len(result)} points (7 days × 96 intervals), range {result['watts'].min():.3f} to {result['watts'].max():.3f}"
    )


def process_marconi():
    """Process Marconi: 60-second sampling, peak 1.698 MW."""
    print("\n" + "=" * 60)
    print("Processing Marconi (60s sampling, 1.698 MW peak)")
    print("=" * 60)

    df = pd.read_csv("Marconi100_power_60_sec.csv")
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Week average
    seconds_per_day = 86400
    seconds_per_week = seconds_per_day * 7
    df["time_of_week"] = df["timestamp_secs"] % seconds_per_week
    weekly_profile = df.groupby("time_of_week")["measured_kW"].mean().reset_index()
    weekly_profile.columns = ["timestamp_s", "kW"]

    # Normalize
    peak_mw = 1.698
    weekly_profile["MW"] = weekly_profile["kW"] / 1000
    weekly_profile["watts"] = weekly_profile["MW"] / peak_mw

    # Resample to 15 min across the week
    max_time = seconds_per_week
    new_timestamps = list(range(0, max_time, 900))
    result = pd.DataFrame(
        {
            "timestamp_s": new_timestamps,
            "timestamp_hr": [t / 3600 for t in new_timestamps],
            "day_of_week": [t // seconds_per_day for t in new_timestamps],
        }
    )

    # Interpolate
    result["watts"] = (
        weekly_profile.set_index("timestamp_s")["watts"]
        .reindex(new_timestamps, method=None)
        .interpolate(method="linear")
        .values
    )

    # Save
    result.to_csv("marconi_week_avg.csv", index=False)

    actual_peak = weekly_profile["MW"].max()
    actual_avg = weekly_profile["MW"].mean()
    print(f"Peak: {actual_peak:.2f} MW ({actual_peak / peak_mw * 100:.1f}% of {peak_mw} MW)")
    print(f"Average: {actual_avg:.2f} MW ({actual_avg / peak_mw * 100:.1f}% of {peak_mw} MW)")
    print(
        f"Output: {len(result)} points (7 days × 96 intervals), range {result['watts'].min():.3f} to {result['watts'].max():.3f}"
    )


def process_perlmutter():
    """Process Perlmutter: 60-second sampling, peak 6.9 MW (already done but included for completeness)."""
    print("\n" + "=" * 60)
    print("Processing Perlmutter (60s sampling, 6.9 MW peak)")
    print("=" * 60)

    df = pd.read_csv("processed_perlmutter.csv", index_col=0)
    print(f"Loaded {len(df)} rows")

    # Week average
    seconds_per_day = 86400
    seconds_per_week = seconds_per_day * 7
    df["time_of_week"] = df["timestamp_secs"] % seconds_per_week
    weekly_profile = df.groupby("time_of_week")["measured_kW"].mean().reset_index()
    weekly_profile.columns = ["timestamp_s", "kW"]

    # Normalize
    peak_mw = 6.9
    weekly_profile["MW"] = weekly_profile["kW"] / 1000
    weekly_profile["watts"] = weekly_profile["MW"] / peak_mw

    # Resample to 15 min across the week
    max_time = seconds_per_week
    new_timestamps = list(range(0, max_time, 900))
    result = pd.DataFrame(
        {
            "timestamp_s": new_timestamps,
            "timestamp_hr": [t / 3600 for t in new_timestamps],
            "day_of_week": [t // seconds_per_day for t in new_timestamps],
        }
    )

    # Interpolate
    result["watts"] = (
        weekly_profile.set_index("timestamp_s")["watts"]
        .reindex(new_timestamps, method=None)
        .interpolate(method="linear")
        .values
    )

    # Save
    result.to_csv("perlmutter_week_avg.csv", index=False)

    actual_peak = weekly_profile["MW"].max()
    actual_avg = weekly_profile["MW"].mean()
    print(f"Peak: {actual_peak:.2f} MW ({actual_peak / peak_mw * 100:.1f}% of {peak_mw} MW)")
    print(f"Average: {actual_avg:.2f} MW ({actual_avg / peak_mw * 100:.1f}% of {peak_mw} MW)")
    print(
        f"Output: {len(result)} points (7 days × 96 intervals), range {result['watts'].min():.3f} to {result['watts'].max():.3f}"
    )


if __name__ == "__main__":
    print("\nBatch processing all cluster load profiles...")
    print("All outputs will have 672 points (15-minute intervals over 7 days)")

    process_cori()
    process_hawk()
    process_lumi()
    process_marconi()
    # process_perlmutter()  # Already processed separately

    print("\n" + "=" * 60)
    print("All files processed successfully!")
    print("=" * 60)
