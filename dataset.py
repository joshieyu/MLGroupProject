import os
import torchaudio
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import get_mel_transform  # Assuming you have this utility function

class AudioDenoiseDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None, target_length=160000, target_time_bins=128):
        # Loading and sorting clean and noisy file paths
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir)])
        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)])
        
        # Transform (Mel transform or any custom transform)
        self.transform = transform or get_mel_transform()
        self.target_length = target_length
        self.target_time_bins = target_time_bins

    def pad_or_crop(self, x):
        """Pads or crops waveform to a fixed length."""
        if x.shape[1] > self.target_length:
            return x[:, :self.target_length]  # Crop
        elif x.shape[1] < self.target_length:
            pad_amount = self.target_length - x.shape[1]
            return F.pad(x, (0, pad_amount))  # Pad
        return x

    def crop_or_pad_spec(self, spec):
        """Pads or crops spectrogram to a fixed time dimension."""
        time_bins = spec.size(-1)
        if time_bins > self.target_time_bins:
            return spec[..., :self.target_time_bins]  # Crop
        elif time_bins < self.target_time_bins:
            pad = self.target_time_bins - time_bins
            return F.pad(spec, (0, pad))  # Pad
        return spec

    def __getitem__(self, idx):
        """Loads a pair of noisy and clean audio files and returns specs and waveforms."""
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_files[idx]

        # Load waveforms
        clean, sr_clean = torchaudio.load(clean_path)
        noisy, sr_noisy = torchaudio.load(noisy_path)

        # Basic validation (optional but good practice)
        # assert sr_clean == sr_noisy, f"Sample rate mismatch: {sr_clean} vs {sr_noisy}"
        # assert clean.shape[0] == noisy.shape[0], "Channel mismatch" # Ensure mono/stereo match

        # Pad or crop waveform to fixed length
        # Store the original waveforms *before* transform for SNR calculation
        clean_waveform_processed = self.pad_or_crop(clean)
        noisy_waveform_processed = self.pad_or_crop(noisy)

        # Convert to Mel spectrogram using the processed waveforms
        clean_spec = self.transform(clean_waveform_processed)  # shape: [1, n_mels, T]
        noisy_spec = self.transform(noisy_waveform_processed)

        # Standardize the time dimension of spectrograms
        clean_spec = self.crop_or_pad_spec(clean_spec)
        noisy_spec = self.crop_or_pad_spec(noisy_spec)

        # Return spectrograms AND the processed waveforms
        return noisy_spec, clean_spec, noisy_waveform_processed, clean_waveform_processed

    def __len__(self):
        """Returns the total number of files."""
        return len(self.clean_files)
