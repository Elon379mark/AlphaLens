from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from agents.deep_learning.patchtst.model import build_patchtst_model

CHECKPOINT_PATH = "models/patchtst/best_model.pt"


def load_best_patchtst(checkpoint_path: str = CHECKPOINT_PATH):
    model = build_patchtst_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def predict_returns(model, dataloader) -> pd.DataFrame:
    """Run inference, extract mean prediction across horizon per sample."""
    results = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(past_values=batch["past_values"], future_values=None)
            preds = outputs.prediction_outputs  # shape: (batch, pred_length, 1)

            for i in range(preds.shape[0]):
                pred_mean = preds[i].float().mean().item()
                actual_mean = batch["future_values"][i].float().mean().item()
                ticker = batch["ticker"][i]
                results.append({"ticker": ticker, "predicted_return": pred_mean, "actual_return": actual_mean})

    return pd.DataFrame(results)


if __name__ == "__main__":
    from agents.deep_learning.patchtst.dataset import prepare_patchtst_data

    print("Preparing data...")
    _, val_dataset = prepare_patchtst_data()
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Loading best checkpoint...")
    model = load_best_patchtst()

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
    predictions_df.to_parquet("outputs/patchtst_predictions.parquet", index=False)
    print("\nSaved to outputs/patchtst_predictions.parquet")

    assert len(predictions_df) > 0, "No predictions generated"
    print("\nPASS: PatchTST inference completed successfully")