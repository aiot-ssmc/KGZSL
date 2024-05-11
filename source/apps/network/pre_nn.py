import torch
import torchaudio
from torch import nn

from apps import SAMPLE_RATE


class Header(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1),
            # nn.GELU(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim, seq_len)
        Return:
            tensor: (batch, seq_len, output_dim)
        """
        return self.net(x).permute(0, 2, 1)


class STFT(nn.Module):
    n_fft = 400
    hop_length = 160

    def __init__(self, hidden_dim):
        super().__init__()
        self.stft_net = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=2.0)
        self.head = Header(self.n_fft // 2 + 1, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, time)
        Return:
            (batch, seq_len, dim)
        """
        spec = self.stft_net(x)
        log_spec = torch.log10(spec + 1e-10) * 10
        return self.head(log_spec)


class Mel(nn.Module):
    n_mels = 80
    n_fft = 400
    hop_length = 160

    def __init__(self, hidden_dim):
        super().__init__()
        self.mel_net = torchaudio.transforms.MelSpectrogram(
            SAMPLE_RATE, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        self.head = Header(self.n_mels, hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_samples)
        mel : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        mel_spec = self.mel_net(x)
        log_spec = torch.log10(mel_spec + 1e-10) * 10
        return self.head(log_spec)


class Conv(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dim = hidden_dim
        self.net = nn.Sequential(
            ConvBlock(1, self.dim, kernel_size=10, stride=5),
            ConvBlock(self.dim, self.dim, kernel_size=3, stride=2),
            ConvBlock(self.dim, self.dim, kernel_size=3, stride=2),
            ConvBlock(self.dim, self.dim, kernel_size=3, stride=2),
            ConvBlock(self.dim, self.dim, kernel_size=3, stride=2),
            ConvBlock(self.dim, self.dim, kernel_size=2, stride=2),
            ConvBlock(self.dim, self.dim, kernel_size=2, stride=2),
            nn.Conv1d(self.dim, self.dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, time)
        Return:
            (batch, seq_len, dim)
        """
        return self.net(x[:, None, :]).permute(0, 2, 1)


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, dropout=0.0):
        super().__init__()
        self.input_channels, self.output_channels = input_channels, output_channels
        self.kernel_size, self.stride = kernel_size, stride
        self.net = nn.Sequential(
            self.make_conv(),
            nn.Dropout(dropout),
            nn.GroupNorm(output_channels, output_channels, affine=True),
            nn.GELU(),
        )

    def make_conv(self):
        conv = nn.Conv1d(self.input_channels, self.output_channels,
                         kernel_size=self.kernel_size, stride=self.stride, bias=False)
        nn.init.kaiming_normal_(conv.weight)
        return conv

    def forward(self, x):
        return self.net(x)
