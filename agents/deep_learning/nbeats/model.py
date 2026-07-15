from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE

def build_nbeats_model(train_dataset: TimeSeriesDataSet) -> NBeats:
    """
    Build N-BEATS interpretable model with trend + seasonality stacks.
    Sizes kept modest given our small synthetic dataset (same reasoning as
    the scaled-down TFT hidden sizes in Chapter 4).
    """
    # DIAGNOSTIC: N-BEATS requires len(reals)==1 and len(flat_categoricals)==0
    # Print actual contents to see what extra field is being added automatically.
    print(f"DIAGNOSTIC — dataset.reals: {train_dataset.reals}")
    print(f"DIAGNOSTIC — dataset.flat_categoricals: {train_dataset.flat_categoricals}")
    print(f"DIAGNOSTIC — dataset.time_varying_unknown_reals: {train_dataset.time_varying_unknown_reals}")
    print(f"DIAGNOSTIC — dataset.target: {train_dataset.target}")

    model = NBeats.from_dataset(
        train_dataset,
        learning_rate=5e-4,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        backcast_loss_ratio=1.0,
        stack_types=["trend", "seasonality"],
        num_blocks=[2, 2],             # smaller than manual's [3,3] — small dataset
        num_block_layers=[3, 3],       # smaller than manual's [4,4]
        widths=[32, 128],              # much smaller than manual's [256,2048] — avoids overfitting on tiny data
        sharing=[True, True],
        expansion_coefficient_lengths=[3, 7],
        loss=SMAPE(),
    )
    return model


if __name__ == "__main__":
    from agents.deep_learning.nbeats.dataset import prepare_nbeats_dataframe, build_nbeats_dataset

    print("Preparing dataset...")
    df = prepare_nbeats_dataframe()
    dataset = build_nbeats_dataset(df)

    print("\nBuilding N-BEATS model from dataset metadata...")
    model = build_nbeats_model(dataset)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel built successfully.")
    print(f"Total parameters: {num_params:,}")

    assert num_params > 0, "Model has zero parameters"
    print(f"\nPASS: N-BEATS model instantiated with {num_params:,} parameters")