from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import mlflow.pytorch

CHECKPOINT_DIR = "models/nbeats"
MAX_EPOCHS = 15


def train_nbeats(model, train_loader, val_loader, max_epochs: int = MAX_EPOCHS) -> pl.Trainer:
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, mode="min")
    checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="nbeats-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("alphalens_nbeats")

    with mlflow.start_run(run_name="NBeats_training"):
        mlflow.log_param("max_epochs", max_epochs)
        trainer.fit(model, train_loader, val_loader)
        if trainer.checkpoint_callback.best_model_path:
            mlflow.log_artifact(trainer.checkpoint_callback.best_model_path)
        mlflow.log_metric(
            "best_val_loss",
            float(checkpoint.best_model_score) if checkpoint.best_model_score else -1.0,
        )

    return trainer


if __name__ == "__main__":
    from agents.deep_learning.nbeats.dataset import prepare_nbeats_dataframe, build_nbeats_dataset, MAX_PREDICTION_LENGTH
    from agents.deep_learning.nbeats.model import build_nbeats_model
    from pytorch_forecasting import TimeSeriesDataSet

    print("Preparing data...")
    df = prepare_nbeats_dataframe()

    training_cutoff = df["time_idx"].max() - MAX_PREDICTION_LENGTH
    train_df = df[df["time_idx"] <= training_cutoff]

    print("Building datasets...")
    train_dataset = build_nbeats_dataset(train_df)
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, df, min_prediction_idx=training_cutoff + 1)

    train_loader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    print("Building model...")
    model = build_nbeats_model(train_dataset)

    print(f"\nStarting training (max {MAX_EPOCHS} epochs)...\n")
    trainer = train_nbeats(model, train_loader, val_loader)

    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score}")

    assert trainer.checkpoint_callback.best_model_path, "No checkpoint was saved"
    print("\nPASS: N-BEATS training completed and checkpoint saved")