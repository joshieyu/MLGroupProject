import os
import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm

def add_noise(waveform, noise_level=0.05):
    noise = noise_level * torch.randn_like(waveform)
    noisy = waveform + noise
    return torch.clamp(noisy, -1.0, 1.0)

def save_librispeech_clean_noisy(
    root="data", subset="train-clean-100", noise_level=0.05, max_samples=None
):
    print("Downloading LibriSpeech...")
    dataset = LIBRISPEECH(root=root, url=subset, download=True)

    clean_dir = os.path.join(root, "clean")
    noisy_dir = os.path.join(root, "noisy")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    print("Generating noisy samples and saving...")

    for i, (waveform, sample_rate, *_ ) in enumerate(tqdm(dataset)):
        if max_samples and i >= max_samples:
            break

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Add noise
        noisy_waveform = add_noise(waveform, noise_level=noise_level)

        # Save files
        torchaudio.save(f"{clean_dir}/sample_{i}.wav", waveform, sample_rate)
        torchaudio.save(f"{noisy_dir}/sample_{i}.wav", noisy_waveform, sample_rate)

    print("Clean and noisy samples saved.")

if __name__ == "__main__":
    save_librispeech_clean_noisy(root="./data", subset="train-clean-100", noise_level=0.05, max_samples=1000)
