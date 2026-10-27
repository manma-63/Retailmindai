import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from src.features import create_features
from src.evaluation import evaluate


EXCLUDED = {
    "date",
    "sales",
    "dataset",
    "frequency"
}


def chronological_split(
    df,
    test_fraction=0.2
):

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    split = int(
        len(df) * (1 - test_fraction)
    )

    if split < 10 or len(df) - split < 1:

        raise ValueError(
            "Not enough data for time-based splitting."
        )

    train = df.iloc[:split].copy()

    test = df.iloc[split:].copy()

    return train, test


def fit_and_predict(data):

    featured = create_features(
        data
    ).dropna().reset_index(drop=True)

    if len(featured) < 20:

        raise ValueError(
            "At least 20 usable observations are required."
        )

    train, test = chronological_split(
        featured
    )

    features = [
        column
        for column in featured.columns
        if column not in EXCLUDED
        and pd.api.types.is_numeric_dtype(
            featured[column]
        )
    ]

    X_train = train[features]

    y_train = train["sales"]

    X_test = test[features]

    y_test = test["sales"]

    try:

        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

        model_name = "XGBoost"

    except ImportError:

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )

        model_name = "RandomForest"

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    predictions = predictions.clip(
        min=0
    )

    output = pd.DataFrame({
        "date": test["date"].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions
    })

    metrics = evaluate(
        y_test.to_numpy(),
        predictions
    )

    return (
        model_name,
        output,
        metrics,
        model
    )