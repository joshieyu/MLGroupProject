import torch
import torchaudio
import torch.nn.functional as F # Import F for padding
from utils import get_mel_transform # Assuming get_mel_transform uses consistent defaults

# Define constants consistent with training/evaluation (based on dataset.py and evaluate.py)
TARGET_SAMPLE_RATE = 16000
TARGET_WAVEFORM_LENGTH = 160000 # From AudioDenoiseDataset default
N_FFT = 1024 # From evaluate.py default/extraction
HOP_LENGTH = 256 # From evaluate.py default/extraction
N_MELS = 96 # From evaluate.py default/extraction

# testing 
def denoise_and_save(model, noisy_path, out_path, sample_rate=TARGET_SAMPLE_RATE): # Add sample_rate parameter
    model.eval()
    waveform, sr = torchaudio.load(noisy_path)

    # 1. Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate # Update sample rate

    # 2. Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Shape: [1, time]

    # 3. Pad or crop waveform to target length (like in dataset.py)
    current_length = waveform.shape[1]
    if current_length > TARGET_WAVEFORM_LENGTH:
        waveform = waveform[:, :TARGET_WAVEFORM_LENGTH]  # Crop
    elif current_length < TARGET_WAVEFORM_LENGTH:
        pad_amount = TARGET_WAVEFORM_LENGTH - current_length
        waveform = F.pad(waveform, (0, pad_amount))  # Pad right

    # 4. Apply Mel transform (ensure get_mel_transform uses consistent N_FFT, HOP_LENGTH, N_MELS)
    # Assuming get_mel_transform uses sample_rate, n_fft=1024, hop_length=256, n_mels=96 by default or via args
    # Remove n_fft and hop_length as they are not accepted by the function signature in utils.py
    transform = get_mel_transform(sample_rate=sr, n_mels=N_MELS)
    mel_spec = transform(waveform)  # Shape: [1, n_mels, time]

    # Add batch dimension → shape: [1, 1, n_mels, time]
    input_tensor = mel_spec.unsqueeze(0)

    # --- Optional: Add padding for time dimension if model requires exact size ---
    target_time_bins = 128 # Example from dataset.py
    current_time_bins = input_tensor.shape[3]
    if current_time_bins < target_time_bins:
        pad_time = target_time_bins - current_time_bins
        input_tensor = F.pad(input_tensor, (0, pad_time)) # Pad time dimension
    elif current_time_bins > target_time_bins:
        input_tensor = input_tensor[:, :, :, :target_time_bins] # Crop time dimension
    # --- End Optional Padding ---


    with torch.no_grad():
        # Ensure model and input are on the same device
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        output = model(input_tensor) # Model processes on device
        # Remove batch dim and move to CPU for inverse transform
        # Output shape is likely [1, 1, n_mels, time]
        output_spec_amp = output.squeeze(0).squeeze(0).cpu() # Shape: [n_mels, time]

    # 5. Convert back to waveform using consistent parameters
    # Use N_FFT consistent with the forward transform
    inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=N_FFT // 2 + 1, n_mels=N_MELS, sample_rate=sr)
    # GriffinLim needs n_fft and hop_length consistent with forward transform
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Apply inverse transforms
    # Ensure output_spec_amp has the correct shape [n_mels, time]
    if output_spec_amp.dim() != 2:
         print(f"Warning: Unexpected output spec dimension before InverseMel: {output_spec_amp.dim()}")
         # Attempt to fix common issues, e.g., extra channel dim
         if output_spec_amp.dim() == 3 and output_spec_amp.shape[0] == 1:
             output_spec_amp = output_spec_amp.squeeze(0)

    spec = inverse_mel(output_spec_amp) # Output shape: [freq, time]
    denoised_waveform = griffin_lim(spec) # Output shape: [time]

    # Add channel dimension back for saving
    denoised_waveform = denoised_waveform.unsqueeze(0) # Shape: [1, time]

    # Save to file
    torchaudio.save(out_path, denoised_waveform, sample_rate=sr)
    print(f"Saved denoised audio to: {out_path}")
