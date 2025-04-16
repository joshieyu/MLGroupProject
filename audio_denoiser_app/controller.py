# audio_denoiser_app/controller.py

import os, sys
# ─── add this block at the very top ──────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import torch
from model import UNetAutoencoder2D       # now resolves to ../model.py
from infer import denoise_and_save         # from ../infer.py

import os
import torch
from model import UNetAutoencoder2D  # from your root project
from infer import denoise_and_save  # the function you already wrote

# Absolute path to your trained weights (adjust as needed)
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "denoising_autoencoder.pt")
)

# Pick device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare the model once
_model = UNetAutoencoder2D().to(DEVICE)
_state = torch.load(MODEL_PATH, map_location=DEVICE)
_model.load_state_dict(_state)
_model.eval()


def denoise_audio(input_path: str, output_path: str) -> str:
    """
    Replaces the placeholder copy with a real denoising pass.
    Returns the path to the denoised file.
    """
    # (optional) ensure output folder exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Run your infer.denoise_and_save under the hood
    # Signature: denoise_and_save(model, noisy_path, out_path, sample_rate=...)
    denoise_and_save(
        _model,
        noisy_path=input_path,
        out_path=output_path,
    )
    return output_path
