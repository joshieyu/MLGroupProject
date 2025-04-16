import os
import torchaudio
import torch
import numpy as np # Make sure numpy is imported if used for SNR avg
import torch.nn.functional as F # Make sure F is imported if used for padding
# Assuming utils contains get_inverse_mel and potentially get_mel_transform
from utils import get_inverse_mel

# --- Helper function for RMS normalization ---
def normalize_rms(waveform, target_rms):
    """Normalizes waveform to have the same RMS as target_rms."""
    current_rms = torch.sqrt(torch.mean(waveform**2))
    if current_rms < 1e-8: # Avoid division by zero for silent waveforms
        return waveform
    scaling_factor = target_rms / current_rms
    return waveform * scaling_factor

# --- Define a default SNR calculation function if not imported ---
def calculate_snr(clean_signal, processed_signal, epsilon=1e-10):
    """Calculates SNR between a clean signal and a processed signal."""
    # Ensure tensors are float and squeezed (as before)
    clean_signal = clean_signal.float().squeeze()
    processed_signal = processed_signal.float().squeeze()

    # Ensure signals have the same length (as before)
    min_len = min(clean_signal.shape[0], processed_signal.shape[0])
    clean_signal = clean_signal[:min_len]
    processed_signal = processed_signal[:min_len]

    # --- Calculate RMS of the clean signal ONCE ---
    # Use the potentially truncated clean_signal for consistent RMS calculation
    clean_rms = torch.sqrt(torch.mean(clean_signal**2))

    # --- Normalize processed_signal to match clean_signal's RMS ---
    # Only normalize if clean_rms is non-zero
    if clean_rms > 1e-8:
        processed_signal_normalized = normalize_rms(processed_signal, clean_rms)
    else:
        processed_signal_normalized = processed_signal # Keep as is if clean is silent

    # --- Calculate SNR using the NORMALIZED processed signal ---
    # The 'signal' is still the original clean signal
    # The 'noise' is the difference between clean and the *normalized* processed signal
    noise = clean_signal - processed_signal_normalized
    signal_power = torch.mean(clean_signal ** 2) # Power of original clean signal
    noise_power = torch.mean(noise ** 2)         # Power of the difference *after* normalization

    snr = 10 * torch.log10(signal_power / (noise_power + epsilon))
    return snr.item()

def pad_or_crop_waveform(x, target_length):
    """Pads or crops waveform tensor to a fixed length."""
    current_length = x.shape[-1]
    if current_length > target_length:
        return x[..., :target_length]
    elif current_length < target_length:
        pad_amount = target_length - current_length
        return F.pad(x, (0, pad_amount))
    return x


# --- Updated Evaluate Function ---
def evaluate(model, dataset, save_dir="denoised_outputs", sample_rate=16000, device='cpu'):
    print(f"--- Evaluating on device: {device} ---", flush=True)
    model.eval()
    # Ensure model is on the correct device (good practice, might be redundant if load_model worked)
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    # --- Determine Transform Parameters (Copy from your previous version) ---
    n_mels = 96  # Default
    n_fft = 1024 # Default
    hop_length = 256 # Default
    try:
        transform_obj = dataset.transform
        if isinstance(transform_obj, torch.nn.Sequential) and len(transform_obj) > 0:
            mel_transform_params = transform_obj[0]
            if hasattr(mel_transform_params, 'n_mels'): n_mels = mel_transform_params.n_mels
            if hasattr(mel_transform_params, 'n_fft'): n_fft = mel_transform_params.n_fft
            if hasattr(mel_transform_params, 'hop_length'): hop_length = mel_transform_params.hop_length
        elif hasattr(transform_obj, 'n_mels'):
             n_mels = transform_obj.n_mels
             n_fft = transform_obj.n_fft
             hop_length = transform_obj.hop_length
        else:
            print("Warning: Could not determine transform parameters. Using defaults.")
    except Exception as e:
        print(f"Warning: Error accessing transform parameters ({e}). Using defaults.")
    print(f"Using parameters for inverse: n_mels={n_mels}, n_fft={n_fft}, hop_length={hop_length}")
    # --- End Transform Parameter Determination ---


    # --- Initialize Transforms OUTSIDE the loop and move to DEVICE ---
    print("Initializing transforms...")
    try:
        # Assuming get_inverse_mel returns torchaudio.transforms.InverseMelScale
        # Pass necessary parameters based on what was determined above
        inverse_mel_transform = get_inverse_mel(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels).to(device)

        # Initialize GriffinLim with matching parameters AND MORE ITERATIONS
        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length, n_iter=64).to(device) # Increased n_iter from default

        print("Transforms initialized and moved to device.")
        # Optional: Check internal tensor device (attribute names might differ)
        # if hasattr(inverse_mel_transform, 'fb'): print(f"InverseMel FB device: {inverse_mel_transform.fb.device}")
        # if hasattr(griffin_lim, 'window'): print(f"GriffinLim window device: {griffin_lim.window.device}")

    except Exception as e:
        print(f"FATAL: Error initializing transforms: {e}")
        return # Stop evaluation if transforms fail

    all_snr_inputs = []
    all_snr_outputs = []

    with torch.no_grad():
        # If your dataset yields more items (like waveforms), unpack them
        # Example: for i, (noisy_spec, clean_spec, noisy_waveform, clean_waveform) in enumerate(dataset):
        for i, data_items in enumerate(dataset):
            # Unpack based on what your dataset actually returns
            # Adjust this based on your AudioDenoiseDataset's __getitem__
            if len(data_items) == 4:
                 noisy_spec, clean_spec, noisy_waveform, clean_waveform = data_items
            elif len(data_items) == 2:
                 noisy_spec, clean_spec = data_items # Or maybe clean_waveform?
                 # If you only get specs, you can't calculate waveform SNR easily here
                 noisy_waveform, clean_waveform = None, None # Indicate missing data
            else:
                 print(f"Warning: Unexpected number of items ({len(data_items)}) yielded by dataset at index {i}. Skipping.")
                 continue

            # --- Move Tensors to DEVICE ---
            noisy_spec = noisy_spec.unsqueeze(0).to(device)  # [1, 1, n_mels, time]

            # Move others if they exist and are needed
            if clean_waveform is not None:
                clean_waveform = clean_waveform.to(device)
            if noisy_waveform is not None:
                noisy_waveform = noisy_waveform.to(device)


            # --- Optional Debug Print ---
            model_param_device = next(model.parameters()).device
            # print(f"Loop {i}: Input Spec Device: {noisy_spec.device}, Model Param Device: {model_param_device}")
            if noisy_spec.device != model_param_device:
                print(f"!!! FATAL MISMATCH at loop {i}: Input on {noisy_spec.device}, Model on {model_param_device}")
                break # Stop if mismatch occurs
            # --- END DEBUG ---

            # --- Model Inference ---
            output_spec = model(noisy_spec)       # Output is on GPU: [1, 1, n_mels, time]
            output_spec_amp = output_spec.squeeze(0)  # Remove batch dim: [1, n_mels, time] -> stays on GPU
            # If model includes channel dim, squeeze(0).squeeze(0) or just squeeze() might be needed
            # Ensure output_spec_amp shape is [n_mels, time] for transforms
            if output_spec_amp.dim() == 3 and output_spec_amp.shape[0] == 1:
                 output_spec_amp = output_spec_amp.squeeze(0) # Now [n_mels, time]

            # Check shape before transform
            # print(f"Loop {i}: Shape of input to InverseMel: {output_spec_amp.shape}, Device: {output_spec_amp.device}")


            # --- Use the pre-initialized transform objects ---
            # The input output_spec_amp is GPU, the transform object is GPU -> result is GPU
            linear_spec = inverse_mel_transform(output_spec_amp)

            # The input linear_spec is GPU, the transform object is GPU -> result is GPU
            denoised_waveform = griffin_lim(linear_spec) # Shape [channels, time] or [time]

            # --- Pad/Crop and Calculate SNR (if possible) ---
            if clean_waveform is not None and noisy_waveform is not None:
                 current_target_length = clean_waveform.shape[-1]
                 # Pad/crop the denoised waveform *before* normalization/SNR calc
                 denoised_waveform_padded = pad_or_crop_waveform(denoised_waveform, current_target_length)

                 # Calculate input SNR (using original calculate_snr logic implicitly)
                 # Note: calculate_snr now normalizes the second argument internally
                 snr_input = calculate_snr(clean_waveform, noisy_waveform)

                 # Calculate output SNR (using original clean and padded denoised)
                 # calculate_snr will normalize denoised_waveform_padded based on clean_waveform
                 snr_output = calculate_snr(clean_waveform, denoised_waveform_padded)

                 snr_improvement = snr_output - snr_input
                 all_snr_inputs.append(snr_input)
                 all_snr_outputs.append(snr_output)
                 print(f"Sample {i}: Input SNR={snr_input:.2f} dB, Output SNR={snr_output:.2f} dB, Improvement={snr_improvement:.2f} dB")
            else:
                 print(f"Sample {i}: Cannot calculate SNR (missing clean/noisy waveform).")


            # --- Save Output Waveform ---
            # Save the *original* (non-normalized) denoised waveform
            filename = os.path.join(save_dir, f"denoised_{i}.wav")
            save_waveform = denoised_waveform # Use the waveform before padding/normalization for saving
            if save_waveform.dim() == 1:
                save_waveform = save_waveform.unsqueeze(0)
            torchaudio.save(filename, save_waveform.cpu(), sample_rate)

        print(f"\nFinished processing {i+1} samples.") # Use i from the loop

    # --- Calculate and print average SNR improvement ---
    if all_snr_outputs and all_snr_inputs:
        avg_snr_input = np.mean(all_snr_inputs)
        avg_snr_output = np.mean(all_snr_outputs)
        avg_improvement = avg_snr_output - avg_snr_input
        print(f"\nAverage Input SNR: {avg_snr_input:.2f} dB")
        print(f"Average Output SNR: {avg_snr_output:.2f} dB")
        print(f"Average SNR Improvement: {avg_improvement:.2f} dB")
    else:
        print("\nCould not calculate average SNR (no samples processed with SNR or SNR calculation failed).")