from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def load_rossmann(
    path=DATA_DIR / "rossmann_train.csv"
):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required = {"Date", "Sales"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Sales"] = pd.to_numeric(
        df["Sales"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Sales"]
    )

    aggregation = {"Sales": "sum"}

    for column in [
        "Promo",
        "SchoolHoliday",
        "Open"
    ]:

        if column in df.columns:

            aggregation[column] = (
                "max"
                if column == "SchoolHoliday"
                else "mean"
            )

    daily = df.groupby(
        "Date",
        as_index=False
    ).agg(aggregation)

    if "StateHoliday" in df.columns:

        holiday = (
            df.assign(
                holiday_flag=df["StateHoliday"]
                .astype(str)
                .str.lower()
                .isin(["a", "b", "c", "1", "2", "3"])
                .astype(int)
            )
            .groupby("Date")["holiday_flag"]
            .max()
        )

        daily["StateHoliday"] = (
            daily["Date"]
            .map(holiday)
            .fillna(0)
            .astype(int)
        )

    else:

        daily["StateHoliday"] = 0

    daily = daily.rename(
        columns={
            "Date": "date",
            "Sales": "sales"
        }
    )

    daily["dataset"] = "Rossmann"

    return daily.sort_values(
        "date"
    ).reset_index(drop=True)


def load_walmart(
    path=DATA_DIR / "walmart_train.csv"
):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if (
        "Weekly Sales" in df.columns
        and "Weekly_Sales" not in df.columns
    ):

        df = df.rename(
            columns={
                "Weekly Sales": "Weekly_Sales"
            }
        )

    required = {
        "Date",
        "Weekly_Sales"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Weekly_Sales"] = pd.to_numeric(
        df["Weekly_Sales"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Weekly_Sales"]
    )

    aggregation = {
        "Weekly_Sales": "sum"
    }

    if "IsHoliday" in df.columns:
        aggregation["IsHoliday"] = "max"

    weekly = df.groupby(
        "Date",
        as_index=False
    ).agg(aggregation)

    weekly = weekly.rename(
        columns={
            "Date": "date",
            "Weekly_Sales": "sales"
        }
    )

    if "IsHoliday" not in weekly.columns:
        weekly["IsHoliday"] = 0

    weekly["IsHoliday"] = (
        weekly["IsHoliday"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
        .astype(int)
    )

    weekly["dataset"] = "Walmart"

    return weekly.sort_values(
        "date"
    ).reset_index(drop=True)