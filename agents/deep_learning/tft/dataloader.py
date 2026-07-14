from typing import Tuple

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

BATCH_SIZE = 32          # smaller than manual's default (64) since our sample dataset is small
NUM_WORKERS = 0          # 0 avoids multiprocessing issues on Windows; safe default
TRAIN_VAL_SPLIT_FRACTION = 0.8  # fraction of time_idx range used for training


def split_train_val(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Split by time_idx (not randomly) — critical for time series: validation
    must come strictly after training in time to avoid leakage.
    Returns (train_df, val_df, training_cutoff).
    """
    max_time_idx = df["time_idx"].max()
    training_cutoff = int(max_time_idx * TRAIN_VAL_SPLIT_FRACTION)

    train_df = df[df["time_idx"] <= training_cutoff]
    val_df = df  # PyTorch Forecasting's predict_from_dataset handles val slicing internally via the cutoff

    return train_df, val_df, training_cutoff


def get_dataloaders(train_dataset: TimeSeriesDataSet, val_dataset: TimeSeriesDataSet) -> Tuple:
    """Build train and validation dataloaders from TimeSeriesDataSet objects."""
    train_loader = train_dataset.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False,
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS, pin_memory=False,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    from agents.deep_learning.tft.dataset import prepare_tft_dataframe, build_tft_dataset

    print("Preparing dataframe...")
    df, feature_cols = prepare_tft_dataframe()

    print("Splitting train/validation by time_idx...")
    train_df, val_df, cutoff = split_train_val(df)
    print(f"Training cutoff (time_idx): {cutoff}")
    print(f"Train rows: {len(train_df)} | Full (val source) rows: {len(val_df)}")

    print("\nBuilding training TimeSeriesDataSet...")
    train_dataset = build_tft_dataset(train_df, feature_cols)

    print("Building validation TimeSeriesDataSet (using training dataset's encoders/normalizers)...")
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_df, predict=True, stop_randomization=True)

    print("\nBuilding dataloaders...")
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

    print(f"\nTrain dataset samples: {len(train_dataset)}")
    print(f"Val dataset samples: {len(val_dataset)}")

    # Pull one batch to sanity-check shapes
    batch = next(iter(train_loader))
    x, y = batch
    print(f"\nSample batch — encoder_target shape: {x['encoder_target'].shape}")
    print(f"Sample batch — decoder_target shape: {x['decoder_target'].shape}")

    assert len(train_dataset) > 0, "Training dataset is empty"
    assert len(val_dataset) > 0, "Validation dataset is empty"
    print("\nPASS: dataloaders built successfully with non-empty train and validation sets")