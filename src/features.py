import numpy as np
import pandas as pd


def create_features(data):

    df = data.copy()

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    dates = pd.to_datetime(df["date"])

    # Calendar features

    df["year"] = dates.dt.year

    df["month"] = dates.dt.month

    df["quarter"] = dates.dt.quarter

    df["day_of_week"] = dates.dt.dayofweek

    df["day_of_year"] = dates.dt.dayofyear

    df["week_of_year"] = (
        dates.dt.isocalendar().week.astype(int)
    )

    # Seasonal features

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    # Missing business indicators

    columns = [
        "Promo",
        "SchoolHoliday",
        "StateHoliday",
        "Open",
        "IsHoliday"
    ]

    for column in columns:

        if column not in df.columns:
            df[column] = 0

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    # Historical sales features

    for lag in [1, 2, 3, 4, 7, 14]:

        df[f"lag_{lag}"] = (
            df["sales"].shift(lag)
        )

    # Rolling statistics use previous sales only

    df["rolling_mean_4"] = (
        df["sales"]
        .shift(1)
        .rolling(4, min_periods=4)
        .mean()
    )

    df["rolling_mean_8"] = (
        df["sales"]
        .shift(1)
        .rolling(8, min_periods=8)
        .mean()
    )

    df["rolling_std_4"] = (
        df["sales"]
        .shift(1)
        .rolling(4, min_periods=4)
        .std()
    )

    return df