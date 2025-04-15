import torch
import torchaudio

def denoise_and_save(model, noisy_path, out_path):
    model.eval()
    # Load the noisy audio file
    waveform, sr = torchaudio.load(noisy_path)
    
    # Ensure stereo is handled correctly (if stereo is needed)
    noisy = waveform.unsqueeze(0)  # Keep stereo channels (if needed for your model)

    # Move model and input to the same device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    noisy = noisy.to(device)

    with torch.no_grad():
        # Denoise the input
        denoised = model(noisy).cpu().squeeze()  # Squeeze to remove unnecessary dimensions

    # Save the denoised output
    torchaudio.save(out_path, denoised.unsqueeze(0), sample_rate=sr)
    print(f"Denoised audio saved to {out_path}")
