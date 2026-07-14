from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss


def build_tft_model(
    train_dataset: TimeSeriesDataSet,
    learning_rate: float = 1e-3,
    hidden_size: int = 16,           # smaller than manual's 64 — appropriate for our small sample dataset
    attention_head_size: int = 2,     # smaller than manual's 4
    dropout: float = 0.1,
    hidden_continuous_size: int = 8,  # smaller than manual's 32
) -> TemporalFusionTransformer:
    """
    Build TFT model from dataset metadata. Sizes are scaled down from the
    manual's defaults since we're training on a small synthetic dataset
    (20 tickers, ~500 days) — full-size hidden dims would overfit badly and
    train unnecessarily slowly on data this small. Scale these up once real,
    larger market data is used.
    """
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=7,  # 7 quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    return model


if __name__ == "__main__":
    from agents.deep_learning.tft.dataset import prepare_tft_dataframe, build_tft_dataset
    from agents.deep_learning.tft.dataloader import split_train_val

    print("Preparing dataframe and dataset...")
    df, feature_cols = prepare_tft_dataframe()
    train_df, val_df, cutoff = split_train_val(df)
    train_dataset = build_tft_dataset(train_df, feature_cols)

    print("\nBuilding TFT model from dataset metadata...")
    model = build_tft_model(train_dataset)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel built successfully.")
    print(f"Total parameters: {num_params:,}")
    print(f"\nModel summary (architecture):")
    print(model)

    assert num_params > 0, "Model has zero parameters — something went wrong"
    print(f"\nPASS: TFT model instantiated with {num_params:,} parameters")