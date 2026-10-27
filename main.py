import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_rossmann, load_walmart
from src.forecasting import fit_and_predict
from src.seasonal_analysis import seasonal_summary
from src.holiday_analysis import holiday_summary


def main():

    parser = argparse.ArgumentParser(
        description="RetailMind AI Sales Forecasting"
    )

    parser.add_argument(
        "--dataset",
        choices=["all", "rossmann", "walmart"],
        default="all"
    )

    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    loaders = {
        "rossmann": load_rossmann,
        "walmart": load_walmart
    }

    if args.dataset == "all":
        selected = loaders
    else:
        selected = {
            args.dataset: loaders[args.dataset]
        }

    scores = []
    predictions = []

    for name, loader in selected.items():

        try:

            data = loader()

            print(f"\nDataset: {name}")
            print(f"Observations: {len(data)}")

            model_name, pred_df, metrics, model = (
                fit_and_predict(data)
            )

            scores.append({
                "Dataset": name,
                "Model": model_name,
                **metrics
            })

            pred_df.insert(0, "dataset", name)
            pred_df.insert(1, "model", model_name)

            predictions.append(pred_df)

            print("Model:", model_name)
            print("Metrics:", metrics)

            seasonal_summary(data)
            holiday_summary(data)

            plt.figure(figsize=(10, 5))

            plt.plot(
                pred_df["date"],
                pred_df["actual"],
                label="Actual Sales"
            )

            plt.plot(
                pred_df["date"],
                pred_df["predicted"],
                label="Predicted Sales"
            )

            plt.title(
                f"{name.title()} Actual vs Predicted Sales"
            )

            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                f"outputs/{name}_actual_vs_predicted.png"
            )

            plt.close()

        except Exception as error:

            print(f"Error processing {name}: {error}")

    if scores:

        pd.DataFrame(scores).to_csv(
            "outputs/model_scores.csv",
            index=False
        )

    if predictions:

        pd.concat(
            predictions,
            ignore_index=True
        ).to_csv(
            "outputs/validation_predictions.csv",
            index=False
        )

    print("\nRetailMind AI execution completed.")


if __name__ == "__main__":
    main()