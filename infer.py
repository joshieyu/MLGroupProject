import torch
import torchaudio
import torch.nn.functional as F
from utils import get_mel_transform

def denoise_and_save(model, noisy_path, out_path):
    model.eval()
    waveform, sr = torchaudio.load(noisy_path)

    # Convert to mono if stereo
    waveform = waveform.mean(dim=0, keepdim=True)  # Shape: [1, time]

    # Apply same transform used in training
    transform = get_mel_transform(sample_rate=sr)  # Should return a MelSpectrogram module
    mel_spec = transform(waveform)  # Shape: [1, n_mels, time]

    target_time_bins = 128
    time_bins = mel_spec.size(-1)
    if time_bins > target_time_bins:
        mel_spec = mel_spec[..., :target_time_bins]
    elif time_bins < target_time_bins:
        pad_amount = target_time_bins - time_bins
        mel_spec = F.pad(mel_spec, (0, pad_amount))

    # Add batch dimension → shape: [1, 1, n_mels, time]
    input_tensor = mel_spec.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor.to(next(model.parameters()).device))
        output = output.squeeze(0).cpu()  # Shape: [1, n_mels, time]

    # Convert back to waveform using Griffin-Lim
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024)
    inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=513, n_mels=output.size(1), sample_rate=sr)

    spec = inverse_mel(output)
    denoised_waveform = griffin_lim(spec)

    # If denoised_waveform is 1D (just samples)
    if denoised_waveform.dim() == 1:
        # Make it [1, samples] for mono audio
        denoised_waveform = denoised_waveform.unsqueeze(0)
    # If it's already 2D, make sure it has the right shape [channels, samples]
    elif denoised_waveform.dim() == 2:
        # Check if shape is [samples, channels] instead of [channels, samples]
        if denoised_waveform.size(0) > denoised_waveform.size(1):
            # Transpose to get [channels, samples]
            denoised_waveform = denoised_waveform.t()

    # Save to file
    torchaudio.save(out_path, denoised_waveform, sample_rate=sr)
    print(f"Saved denoised audio to: {out_path}")
