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

    # Add batch dimension → shape: [1, 1, n_mels, time]
    input_tensor = mel_spec.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor.to(next(model.parameters()).device))
        output = output.squeeze(0).cpu()  # Shape: [1, n_mels, time]

    # Convert back to waveform using Griffin-Lim
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024)
    inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=512, n_mels=output.size(0), sample_rate=sr)

    spec = inverse_mel(output)
    denoised_waveform = griffin_lim(spec)

    # Save to file
    torchaudio.save(out_path, denoised_waveform.unsqueeze(0), sample_rate=sr)
    print(f"Saved denoised audio to: {out_path}")
