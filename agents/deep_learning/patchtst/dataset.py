from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CONTEXT_LENGTH = 60   # lookback window PatchTST sees
PRED_LENGTH = 21       # forecast horizon, matches other chapters


class PatchTSTDataset(Dataset):
    """
    Plain PyTorch Dataset producing sliding windows of (past_values, future_values)
    per ticker. PatchTST (via HuggingFace) expects raw tensors directly, unlike
    pytorch-forecasting's TimeSeriesDataSet used for TFT/N-BEATS.
    """
    def __init__(self, series_by_ticker: dict, context_length: int = CONTEXT_LENGTH, pred_length: int = PRED_LENGTH):
        self.context_length = context_length
        self.pred_length = pred_length
        self.samples = []  # list of (ticker, past_values, future_values)

        for ticker, series in series_by_ticker.items():
            series = np.asarray(series, dtype=np.float32)
            total_window = context_length + pred_length
            for start in range(0, len(series) - total_window + 1):
                past = series[start : start + context_length]
                future = series[start + context_length : start + total_window]
                self.samples.append((ticker, past, future))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ticker, past, future = self.samples[idx]
        return {
            "past_values": torch.tensor(past, dtype=torch.float32).unsqueeze(-1),   # (context_length, 1 channel)
            "future_values": torch.tensor(future, dtype=torch.float32).unsqueeze(-1),  # (pred_length, 1 channel)
            "ticker": ticker,
        }


def prepare_patchtst_data(
    prices_path: str = "data/processed/sample_prices.parquet",
    context_length: int = CONTEXT_LENGTH,
    pred_length: int = PRED_LENGTH,
) -> Tuple[PatchTSTDataset, PatchTSTDataset]:
    """
    Load prices, compute daily returns per ticker, split into train/val by
    time (last pred_length*3 days held out for validation), build datasets.
    """
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["daily_return"] = prices.groupby("ticker")["adj_close"].pct_change()
    prices = prices.dropna(subset=["daily_return"])

    series_by_ticker = {
        ticker: group["daily_return"].values
        for ticker, group in prices.groupby("ticker")
    }

    # Time-based split: last portion of each series held out for validation
    val_holdout = pred_length * 3
    train_series = {t: s[:-val_holdout] for t, s in series_by_ticker.items() if len(s) > (context_length + pred_length + val_holdout)}
    val_series = {t: s[-(context_length + pred_length + val_holdout):] for t, s in series_by_ticker.items() if len(s) > (context_length + pred_length + val_holdout)}

    train_dataset = PatchTSTDataset(train_series, context_length, pred_length)
    val_dataset = PatchTSTDataset(val_series, context_length, pred_length)

    return train_dataset, val_dataset


if __name__ == "__main__":
    print("Preparing PatchTST datasets...")
    train_dataset, val_dataset = prepare_patchtst_data()

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    sample = train_dataset[0]
    print(f"\nSample — ticker: {sample['ticker']}")
    print(f"past_values shape: {sample['past_values'].shape}")
    print(f"future_values shape: {sample['future_values'].shape}")

    assert len(train_dataset) > 0, "Train dataset is empty"
    assert len(val_dataset) > 0, "Val dataset is empty"
    assert sample["past_values"].shape == (CONTEXT_LENGTH, 1), f"Unexpected past_values shape: {sample['past_values'].shape}"
    assert sample["future_values"].shape == (PRED_LENGTH, 1), f"Unexpected future_values shape: {sample['future_values'].shape}"

    print("\nPASS: PatchTST datasets prepared and built successfully")