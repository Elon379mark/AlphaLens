from transformers import PatchTSTConfig, PatchTSTForPrediction

from agents.deep_learning.patchtst.dataset import CONTEXT_LENGTH, PRED_LENGTH

PATCH_LENGTH = 8    # smaller than manual's 16 — appropriate for our shorter 60-day context
PATCH_STRIDE = 4    # smaller than manual's 8, correspondingly


def build_patchtst_model(num_input_channels: int = 1) -> PatchTSTForPrediction:
    """
    Build PatchTST via HuggingFace Transformers.
    num_input_channels=1 since we're forecasting a single return series per ticker
    (univariate), matching N-BEATS's scope rather than TFT's multivariate setup.
    """
    config = PatchTSTConfig(
        num_input_channels=num_input_channels,
        context_length=CONTEXT_LENGTH,
        patch_length=PATCH_LENGTH,
        patch_stride=PATCH_STRIDE,
        prediction_length=PRED_LENGTH,
        d_model=32,              # smaller than manual's 128 — small dataset
        num_attention_heads=4,    # smaller than manual's 16
        num_hidden_layers=2,      # smaller than manual's 3
        ffn_dim=64,               # smaller than manual's 256
        dropout=0.2,
        head_dropout=0.2,
        pooling_type="mean",
        channel_attention=False,
        scaling="std",
        loss="mse",
        pre_norm=True,
    )
    return PatchTSTForPrediction(config)


if __name__ == "__main__":
    print("Building PatchTST model...")
    model = build_patchtst_model()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel built successfully.")
    print(f"Total parameters: {num_params:,}")

    assert num_params > 0, "Model has zero parameters"
    print(f"\nPASS: PatchTST model instantiated with {num_params:,} parameters")