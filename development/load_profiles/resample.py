"""
Resample year-long cluster data to day-averaged load profile.

This script takes a full year of cluster power data and:
1. Groups data by time-of-day across all days
2. Averages each time point across all days to get a representative daily profile
3. Normalizes to a peak value of 6.9 MW
4. Outputs in the desired format with timestamp_s, watts, and timestamp_hr
"""

import pandas as pd


def process_cluster_data(input_file, output_file, peak_mw=6.9, resample_interval="15min"):
    """
    Process year-long cluster data into day-averaged normalized profile.

    Parameters
    ----------
    input_file : str
        Path to input CSV with columns: timestamp_secs, measured_kW, days
    output_file : str
        Path to output CSV
    peak_mw : float
        Peak power in MW for normalization (default: 6.9)
    resample_interval : str
        Resampling interval (default: '15min')
    """

    # Read the data
    df = pd.read_csv(input_file, index_col=0)

    # Convert timestamp_secs to datetime, using days column to get full timestamps
    # Assuming the data starts at some epoch
    df["datetime"] = pd.to_datetime(df["timestamp_secs"], unit="s")

    # Alternatively, if we want to use days column:
    # df['datetime'] = pd.to_timedelta(df['days'], unit='D') + pd.Timestamp('2000-01-01')

    # Set datetime as index
    df = df.set_index("datetime")

    # Extract time of day (seconds since midnight)
    df["time_of_day"] = df.index.hour * 3600 + df.index.minute * 60 + df.index.second

    # Group by time_of_day and average across all days
    # This gives us the mean load at each time point across the entire year
    daily_avg = df.groupby("time_of_day")["measured_kW"].mean()

    # Resample to desired interval if needed
    # First, create a proper time index for the daily profile
    daily_df = pd.DataFrame({"timestamp_s": daily_avg.index, "kW": daily_avg.values})

    # Convert to watts
    daily_df["watts"] = daily_df["kW"] * 1000

    # Normalize to peak value
    peak_watts = peak_mw * 1e6  # Convert MW to watts
    current_peak = daily_df["watts"].max()
    daily_df["watts_normalized"] = daily_df["watts"] / current_peak * peak_watts

    # Convert watts to fractional units (normalized to peak)
    # Based on your example output showing values around 0.9
    daily_df["watts_fraction"] = daily_df["watts_normalized"] / peak_watts

    # Create timestamp_hr column
    daily_df["timestamp_hr"] = daily_df["timestamp_s"] / 3600

    # Resample if needed
    if resample_interval:
        # Create a datetime index for resampling
        temp_df = daily_df.copy()
        temp_df["temp_datetime"] = pd.to_timedelta(temp_df["timestamp_s"], unit="s")
        temp_df = temp_df.set_index("temp_datetime")

        # Resample to desired interval
        resampled = temp_df.resample(resample_interval).mean()

        # Recreate columns
        daily_df = pd.DataFrame(
            {
                "timestamp_s": resampled.index.total_seconds(),
                "watts": resampled["watts_fraction"],
                "timestamp_hr": resampled.index.total_seconds() / 3600,
            }
        )
    else:
        # Use the fractional representation
        daily_df = daily_df[["timestamp_s", "watts_fraction", "timestamp_hr"]].copy()
        daily_df.columns = ["timestamp_s", "watts", "timestamp_hr"]

    # Reset index
    daily_df = daily_df.reset_index(drop=True)

    # Save to output file
    daily_df.to_csv(output_file, index=True)

    print(f"Processed data saved to {output_file}")
    print(f"Original peak: {current_peak / 1e6:.2f} MW")
    print(f"Normalized peak: {peak_mw} MW")
    print(f"Output shape: {daily_df.shape}")
    print("\nFirst few rows:")
    print(daily_df.head(10))

    return daily_df


def simple_day_average(input_file, output_file, peak_mw=6.9, resample_seconds=900):
    """
    Simplified version: average by time-of-day without fancy resampling.

    This directly implements:
    1. Group by time-of-day (seconds since midnight)
    2. Average across all days
    3. Normalize to peak value
    4. Resample to specified interval (default: 900s = 15min)

    Parameters
    ----------
    input_file : str
        Path to input CSV with columns: timestamp_secs, measured_kW, days
    output_file : str
        Path to output CSV
    peak_mw : float
        Peak power in MW for normalization (default: 6.9)
    resample_seconds : int or None
        Resampling interval in seconds (default: 900 = 15 minutes)
        Set to None to keep original sampling rate
    """

    # Read the data
    df = pd.read_csv(input_file, index_col=0)

    # Calculate seconds since midnight for each timestamp
    # Assuming timestamp_secs represents seconds from start of year
    seconds_per_day = 86400
    df["time_of_day"] = df["timestamp_secs"] % seconds_per_day

    # Group by time_of_day and average
    daily_profile = df.groupby("time_of_day")["measured_kW"].mean().reset_index()
    daily_profile.columns = ["timestamp_s", "kW"]

    # Convert to MW and normalize to fractional units where peak_mw = 1.0
    daily_profile["MW"] = daily_profile["kW"] / 1000

    # Normalize: divide by peak_mw so that peak_mw MW = 1.0
    # This means if data is at 2.5 MW and peak is 6.9 MW, output is 2.5/6.9 ≈ 0.36
    daily_profile["watts"] = daily_profile["MW"] / peak_mw

    # Add timestamp_hr
    daily_profile["timestamp_hr"] = daily_profile["timestamp_s"] / 3600

    # Resample to desired interval if specified
    if resample_seconds is not None:
        # Create bins for resampling
        max_time = 86400  # seconds in a day
        new_timestamps = pd.Series(range(0, max_time, resample_seconds))

        # Create a DataFrame for interpolation
        # We'll use linear interpolation to get values at exact 15-min intervals
        result = pd.DataFrame({
            "timestamp_s": new_timestamps,
            "timestamp_hr": new_timestamps / 3600
        })

        # Interpolate watts values to the new timestamps
        result["watts"] = pd.Series(
            daily_profile.set_index("timestamp_s")["watts"].reindex(
                new_timestamps, method=None
            ).interpolate(method="linear")
        ).values

        # Handle edge case: if we don't have data at t=0, use nearest value
        if pd.isna(result["watts"].iloc[0]):
            result["watts"].iloc[0] = daily_profile["watts"].iloc[0]
    else:
        # Select final columns without resampling
        result = daily_profile[["timestamp_s", "watts", "timestamp_hr"]].copy()

    # Save
    result.to_csv(output_file, index=True)

    # Calculate actual peak and average for reporting
    actual_peak_mw = daily_profile["MW"].max()
    actual_avg_mw = daily_profile["MW"].mean()

    print(f"Processed data saved to {output_file}")
    print(f"Actual data peak: {actual_peak_mw:.2f} MW ({actual_peak_mw/peak_mw*100:.1f}% of {peak_mw} MW)")
    print(f"Actual data average: {actual_avg_mw:.2f} MW ({actual_avg_mw/peak_mw*100:.1f}% of {peak_mw} MW)")
    print(f"Output points: {len(result)}")
    print(f"Output range: {result['watts'].min():.3f} to {result['watts'].max():.3f}")
    print("\nFirst few rows:")
    print(result.head(10))

    return result


if __name__ == "__main__":
    # Example usage
    input_file = "processed_perlmutter.csv"
    output_file = "day_averaged_profile.csv"

    # Use the simple version with 15-minute resampling (900 seconds)
    result = simple_day_average(input_file, output_file, peak_mw=6.9, resample_seconds=900)
