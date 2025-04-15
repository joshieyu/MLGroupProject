import torchaudio.transforms as T
import torch

def get_mel_transform(sample_rate=16000):
    return torch.nn.Sequential(
        T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        ),
        T.AmplitudeToDB()  # Log scale helps learning
    )
def get_inverse_mel(sample_rate=16000, n_fft=1024, n_mels=128):
    return T.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate
    )
