from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import mlflow.pytorch

CHECKPOINT_DIR = "models/tft"
MAX_EPOCHS = 15  # kept small given tiny synthetic dataset — real data would warrant more


def train_tft(model, train_loader, val_loader, max_epochs: int = MAX_EPOCHS) -> pl.Trainer:
    """
    Train the TFT model with early stopping and checkpointing.
    Logs to MLflow. Returns the fitted Trainer.
    """
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, mode="min",
    )
    checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",  # explicit CPU — matches our CPU-only torch install
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("alphalens_tft")

    with mlflow.start_run(run_name="TFT_training"):
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("hidden_size", model.hparams.hidden_size)
        mlflow.log_param("learning_rate", model.hparams.learning_rate)

        trainer.fit(model, train_loader, val_loader)

        if trainer.checkpoint_callback.best_model_path:
            mlflow.log_artifact(trainer.checkpoint_callback.best_model_path)
        mlflow.log_metric("best_val_loss", float(checkpoint.best_model_score) if checkpoint.best_model_score else -1.0)

    return trainer


if __name__ == "__main__":
    from agents.deep_learning.tft.dataset import prepare_tft_dataframe, build_tft_dataset
    from agents.deep_learning.tft.dataloader import split_train_val, get_dataloaders
    from agents.deep_learning.tft.model import build_tft_model
    from pytorch_forecasting import TimeSeriesDataSet

    print("Preparing data...")
    df, feature_cols = prepare_tft_dataframe()
    train_df, val_df, cutoff = split_train_val(df)

    print("Building datasets...")
    train_dataset = build_tft_dataset(train_df, feature_cols)
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_df, predict=True, stop_randomization=True)

    print("Building dataloaders...")
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

    print("Building model...")
    model = build_tft_model(train_dataset)

    print(f"\nStarting training (max {MAX_EPOCHS} epochs, early stopping on val_loss)...")
    print("This may take several minutes on CPU — progress bars will show per-epoch status.\n")

    trainer = train_tft(model, train_loader, val_loader)

    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score}")
    print(f"Stopped at epoch: {trainer.current_epoch}")

    assert trainer.checkpoint_callback.best_model_path, "No checkpoint was saved — training may have failed silently"
    print("\nPASS: TFT training completed and checkpoint saved")