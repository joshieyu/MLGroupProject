# import torch
# import torchaudio
# import os
# from utils import get_mel_transform, get_inverse_mel

# def evaluate(model, dataset, save_dir="denoised_outputs", sample_rate=16000):
#     model.eval()
#     os.makedirs(save_dir, exist_ok=True)

#     inverse_mel = get_inverse_mel(sample_rate=sample_rate)
#     griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024)

#     with torch.no_grad():
#         for i, (noisy_spec, _) in enumerate(dataset):
#             noisy_spec = noisy_spec.unsqueeze(0)  # [1, 1, n_mels, time]
#             output_spec = model(noisy_spec)       # [1, 1, n_mels, time]
#             mel = output_spec.squeeze(0)  # [n_mels, time]

#             # Invert back to waveform
#             linear_spec = inverse_mel(mel)
#             waveform = griffin_lim(linear_spec)

#             # Save output
#             filename = os.path.join(save_dir, f"denoised_{i}.wav")
#             torchaudio.save(filename, waveform.unsqueeze(0), sample_rate)
#             print(f"🔊 Saved: {filename}")
import os
import torchaudio
import torch
from utils import get_mel_transform, get_inverse_mel  # Assuming these exist

def evaluate(model, dataset, save_dir="denoised_outputs", sample_rate=16000, device='cpu'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024)

    with torch.no_grad():
        for i, (noisy_spec, _) in enumerate(dataset):
            noisy_spec = noisy_spec.unsqueeze(0).to(device)  # [1, 1, n_mels, time] and move to device
            output_spec = model(noisy_spec)  # [1, 1, n_mels, time]
            mel = output_spec.squeeze(0)  # Only remove batch dimension, should be [n_mels, time]

            # Optionally clip output Mel spectrogram to ensure valid range
            mel = torch.clamp(mel, 0, 1)

            # Invert back to a linear spectrogram (assuming the `get_inverse_mel` is correct)
            linear_spec = get_inverse_mel()(mel)  # This is the Mel-to-linear inversion

            # Convert the linear spectrogram to waveform using Griffin-Lim
            waveform = griffin_lim(linear_spec)

            # Ensure waveform has the shape [1, time] for saving (adding the batch dimension)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # [1, time] if it was 1D

            # Save output as a waveform with shape [1, time]
            filename = os.path.join(save_dir, f"denoised_{i}.wav")
            torchaudio.save(filename, waveform, sample_rate)  # [1, time] shape
            print(f"Saved: {filename}")


