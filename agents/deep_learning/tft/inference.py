import glob
from pathlib import Path

import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

CHECKPOINT_DIR = "models/tft"


def find_best_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> str:
    """Find the most recently saved .ckpt file in the checkpoint directory."""
    ckpts = glob.glob(f"{checkpoint_dir}/*.ckpt")
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    latest = max(ckpts, key=lambda p: Path(p).stat().st_mtime)
    return latest


def load_best_tft(checkpoint_path: str) -> TemporalFusionTransformer:
    """Load a trained TFT model from checkpoint."""
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


def predict_returns(model: TemporalFusionTransformer, dataloader) -> pd.DataFrame:
    """
    Run inference and extract mean prediction across the horizon per ticker.
    Returns DataFrame with columns: [ticker, predicted_return, actual_return].
    """
    with torch.no_grad():
        raw_preds = model.predict(dataloader, mode="prediction", return_x=True, return_y=True, return_index=True)

    predictions = raw_preds.output   # shape: (n_samples, prediction_length)
    index_df = raw_preds.index       # DataFrame with time_idx, ticker per sample
    actuals = raw_preds.y[0]         # y is a tuple; first element is the target tensor, shape matches predictions

    results = []
    for i in range(predictions.shape[0]):
        ticker = index_df.iloc[i]["ticker"]
        pred_mean = predictions[i].float().mean().item()
        actual_mean = actuals[i].float().mean().item()
        results.append({
            "ticker": ticker,
            "predicted_return": pred_mean,
            "actual_return": actual_mean,
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    from agents.deep_learning.tft.dataset import prepare_tft_dataframe, build_tft_dataset
    from agents.deep_learning.tft.dataloader import split_train_val, get_dataloaders

    print("Preparing data...")
    df, feature_cols = prepare_tft_dataframe()
    train_df, val_df, cutoff = split_train_val(df)

    train_dataset = build_tft_dataset(train_df, feature_cols)
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_df, predict=True, stop_randomization=True)
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

    print("\nFinding best checkpoint...")
    ckpt_path = find_best_checkpoint()
    print(f"Loading: {ckpt_path}")

    model = load_best_tft(ckpt_path)
    print("Model loaded.")

    print("\nRunning inference on validation set...")
    predictions_df = predict_returns(model, val_loader)

    print(f"\nPredictions shape: {predictions_df.shape}")
    print(predictions_df.head(10))

    # Directional accuracy: does predicted sign match actual sign?
    predictions_df["direction_correct"] = (
        (predictions_df["predicted_return"] > 0) == (predictions_df["actual_return"] > 0)
    )
    directional_accuracy = predictions_df["direction_correct"].mean()
    print(f"\nDirectional accuracy: {directional_accuracy:.1%}")
    print("(On synthetic random-walk data, expect this close to 50% — no real signal to predict)")

    Path("outputs").mkdir(exist_ok=True)
    predictions_df.to_parquet("outputs/tft_predictions.parquet", index=False)
    print("\nSaved to outputs/tft_predictions.parquet")

    assert len(predictions_df) > 0, "No predictions generated"
    print("\nPASS: TFT inference completed successfully")