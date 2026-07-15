import glob
from pathlib import Path

import pandas as pd
import torch
from pytorch_forecasting import NBeats

CHECKPOINT_DIR = "models/nbeats"


def find_best_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> str:
    ckpts = glob.glob(f"{checkpoint_dir}/*.ckpt")
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    return max(ckpts, key=lambda p: Path(p).stat().st_mtime)


def load_best_nbeats(checkpoint_path: str) -> NBeats:
    model = NBeats.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


def predict_returns(model: NBeats, dataloader) -> pd.DataFrame:
    with torch.no_grad():
        raw_preds = model.predict(dataloader, mode="prediction", return_x=True, return_y=True, return_index=True)

    predictions = raw_preds.output
    index_df = raw_preds.index
    actuals = raw_preds.y[0]

    results = []
    for i in range(predictions.shape[0]):
        ticker = index_df.iloc[i]["ticker"]
        pred_mean = predictions[i].float().mean().item()
        actual_mean = actuals[i].float().mean().item()
        results.append({"ticker": ticker, "predicted_return": pred_mean, "actual_return": actual_mean})

    return pd.DataFrame(results)


if __name__ == "__main__":
    from agents.deep_learning.nbeats.dataset import prepare_nbeats_dataframe, build_nbeats_dataset, MAX_PREDICTION_LENGTH
    from pytorch_forecasting import TimeSeriesDataSet

    print("Preparing data...")
    df = prepare_nbeats_dataframe()
    training_cutoff = df["time_idx"].max() - MAX_PREDICTION_LENGTH
    train_df = df[df["time_idx"] <= training_cutoff]

    train_dataset = build_nbeats_dataset(train_df)
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, df, min_prediction_idx=training_cutoff + 1)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    print("Finding best checkpoint...")
    ckpt_path = find_best_checkpoint()
    print(f"Loading: {ckpt_path}")
    model = load_best_nbeats(ckpt_path)

    print("\nRunning inference...")
    predictions_df = predict_returns(model, val_loader)

    print(f"\nPredictions shape: {predictions_df.shape}")
    print(predictions_df.head(10))

    predictions_df["direction_correct"] = (
        (predictions_df["predicted_return"] > 0) == (predictions_df["actual_return"] > 0)
    )
    print(f"\nDirectional accuracy: {predictions_df['direction_correct'].mean():.1%}")
    print("(On synthetic random-walk data, expect close to 50%)")

    Path("outputs").mkdir(exist_ok=True)
    predictions_df.to_parquet("outputs/nbeats_predictions.parquet", index=False)
    print("\nSaved to outputs/nbeats_predictions.parquet")

    assert len(predictions_df) > 0, "No predictions generated"
    print("\nPASS: N-BEATS inference completed successfully")