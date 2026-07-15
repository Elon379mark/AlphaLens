from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import mlflow

CHECKPOINT_DIR = "models/patchtst"
MAX_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


def evaluate_patchtst(model, val_loader, device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                past_values=batch["past_values"].to(device),
                future_values=batch["future_values"].to(device),
            )
            total_loss += outputs.loss.item()
            n_batches += 1
    return total_loss / n_batches if n_batches > 0 else float("inf")


def train_patchtst(model, train_loader, val_loader, max_epochs: int = MAX_EPOCHS) -> str:
    """
    Plain PyTorch training loop (HuggingFace PatchTSTForPrediction is not a
    LightningModule, so we don't use lightning.pytorch.Trainer here — this
    is a deliberate, correct difference from the TFT/N-BEATS chapters, not
    an inconsistency).
    Returns path to the best saved checkpoint.
    """
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    best_ckpt_path = f"{CHECKPOINT_DIR}/best_model.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=max_epochs)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("alphalens_patchtst")

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    with mlflow.start_run(run_name="PatchTST_training"):
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("learning_rate", LEARNING_RATE)

        first_batch_diagnosed = False
        for epoch in range(max_epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(
                    past_values=batch["past_values"].to(device),
                    future_values=batch["future_values"].to(device),
                )
                outputs.loss.backward()

                if not first_batch_diagnosed:
                    total_grad_norm = 0.0
                    num_params_with_grad = 0
                    num_params_total = 0
                    for name, p in model.named_parameters():
                        num_params_total += 1
                        if p.grad is not None:
                            num_params_with_grad += 1
                            total_grad_norm += p.grad.norm().item()
                        else:
                            print(f"DIAGNOSTIC — no gradient on: {name}")
                    print(f"DIAGNOSTIC — loss value: {outputs.loss.item()}")
                    print(f"DIAGNOSTIC — params with gradient: {num_params_with_grad}/{num_params_total}")
                    print(f"DIAGNOSTIC — total gradient norm: {total_grad_norm}")
                    print(f"DIAGNOSTIC — requires_grad on model params: {all(p.requires_grad for p in model.parameters())}")
                    first_batch_diagnosed = True

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                

            val_loss = evaluate_patchtst(model, val_loader, device)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            print(f"Epoch {epoch+1}/{max_epochs}: val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        if Path(best_ckpt_path).exists():
            mlflow.log_artifact(best_ckpt_path)

    return best_ckpt_path


if __name__ == "__main__":
    from agents.deep_learning.patchtst.dataset import prepare_patchtst_data
    from agents.deep_learning.patchtst.model import build_patchtst_model

    print("Preparing data...")
    train_dataset, val_dataset = prepare_patchtst_data()
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nBuilding model...")
    model = build_patchtst_model()

    print(f"\nStarting training (max {MAX_EPOCHS} epochs, early stopping patience=5)...\n")
    best_ckpt = train_patchtst(model, train_loader, val_loader)

    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Best checkpoint: {best_ckpt}")

    assert Path(best_ckpt).exists(), "No checkpoint was saved"
    print("\nPASS: PatchTST training completed and checkpoint saved")