import os
import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm
import shutil # Import shutil for potentially cleaning old directories

def add_noise(waveform, noise_level=0.05):
    """Adds Gaussian noise to a waveform.

    Args:
        waveform (Tensor): The input clean waveform.
        noise_level (float): A factor controlling the noise intensity.
                             Lower values mean less noise relative to the signal.
                             Specifically, it scales the noise std dev relative
                             to the signal std dev.
    Returns:
        Tensor: The waveform with added noise.
    """
    # Calculate the standard deviation of the clean signal
    signal_std = torch.std(waveform)

    # Generate noise with the same shape as the waveform
    noise = torch.randn_like(waveform)

    # Calculate the standard deviation of the generated noise
    noise_std = torch.std(noise)

    # Scale the noise standard deviation
    # Target noise std = signal_std * noise_level
    # Scaling factor = target_noise_std / current_noise_std
    if noise_std > 1e-8: # Avoid division by zero if noise is somehow flat
        scaled_noise = noise * (signal_std * noise_level / noise_std)
    else:
        scaled_noise = torch.zeros_like(waveform) # Or handle as an error

    # Add the scaled noise to the original waveform
    noisy_waveform = waveform + scaled_noise
    return noisy_waveform

def save_librispeech_clean_noisy(
    root="data", subset="train-clean-100", noise_levels=[0.05], max_samples=None, clean_first=True
):
    """
    Downloads LibriSpeech subset, saves clean samples, and generates noisy versions
    at multiple specified noise levels in separate directories.

    Args:
        root (str): Root directory for data.
        subset (str): LibriSpeech subset URL name.
        noise_levels (list): A list of noise levels (floats) to generate.
        max_samples (int, optional): Maximum number of samples to process. Defaults to None (all).
        clean_first (bool): If True, removes existing clean/noisy directories before starting.
    """
    print("Downloading LibriSpeech...")
    # Download to a temporary subfolder to avoid mixing with processed data if root='.'
    download_root = os.path.join(root, "_download")
    os.makedirs(download_root, exist_ok=True) # Ensure download directory exists BEFORE dataset init
    dataset = LIBRISPEECH(root=download_root, url=subset, download=True)

    clean_dir = os.path.join(root, "clean")
    noisy_dirs = {level: os.path.join(root, f"noisy_{level:.3f}".replace('.', '_')) for level in noise_levels} # Create dict for noisy dirs

    if clean_first:
        print("Cleaning existing directories...")
        if os.path.exists(clean_dir):
            shutil.rmtree(clean_dir)
        for noisy_dir in noisy_dirs.values():
            if os.path.exists(noisy_dir):
                shutil.rmtree(noisy_dir)

    print(f"Ensuring directories exist...")
    os.makedirs(clean_dir, exist_ok=True)
    for noisy_dir in noisy_dirs.values():
        os.makedirs(noisy_dir, exist_ok=True)

    print("Generating clean and noisy samples...")

    count = 0
    # Use tqdm correctly with enumerate
    # Iterate through the downloaded dataset
    for i, (waveform, sample_rate, *_) in enumerate(tqdm(dataset, desc="Processing samples")):
        if max_samples and count >= max_samples:
            break

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # --- Save clean file once ---
        clean_filename = os.path.join(clean_dir, f"sample_{count}.wav")
        torchaudio.save(clean_filename, waveform, sample_rate)

        # --- Generate and save noisy files for each level ---
        for level in noise_levels:
            noisy_waveform = add_noise(waveform, noise_level=level)
            noisy_filename = os.path.join(noisy_dirs[level], f"sample_{count}.wav")
            torchaudio.save(noisy_filename, noisy_waveform, sample_rate)

        count += 1

    # Clean up download directory (optional)
    # print("Cleaning up download directory...")
    # shutil.rmtree(download_root)

    print(f"\nClean and noisy samples saved ({count} total samples processed).")
    print(f"Clean directory: {clean_dir}")
    print("Noisy directories:")
    for level, path in noisy_dirs.items():
        print(f" - Level {level:.3f}: {path}")


if __name__ == "__main__":
    # Define the desired noise levels
    noise_levels_to_generate = [0.01, 0.05, 0.1, 0.2, 0.5] # Example levels

    # Generate 100 samples for each specified noise level
    save_librispeech_clean_noisy(
        root="./data",
        subset="train-clean-100",
        noise_levels=noise_levels_to_generate,
        max_samples=100,
        clean_first=True # Set to False if you want to add levels without deleting existing ones
    )
