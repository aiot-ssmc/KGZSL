# refs https://github.com/KinWaiCheuk/nnAudio.git
import abc
import numpy
import scipy
import torch
from torch import nn

from .. import t2n, EPS


def pad_center(data: numpy.ndarray, expected_size: int, axis=-1, mode='constant', **kwargs):
    o_size = data.shape[axis]
    left_pad_size = (expected_size - o_size) // 2
    right_pad_size = expected_size - o_size - left_pad_size
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (left_pad_size, right_pad_size)
    return numpy.pad(data, lengths, mode=mode, **kwargs)


def overlap_add(x: torch.Tensor, stride):
    n_fft = x.shape[1]
    n_len = x.shape[-1]
    output_len = n_fft + stride * (n_len - 1)
    return torch.nn.functional.fold(x, (1, output_len), kernel_size=(1, n_fft), stride=stride).flatten(1)


class Kernel(nn.Module):
    def __init__(self, n_fft, window_name, win_length, trainable):
        super().__init__()
        self.n_fft = n_fft
        window_mask = self.get_window_mask(window_name, win_length)
        kernel_sin, kernel_cos = self.create_fourier_kernels()
        if trainable:
            self.kernel_sin = nn.Parameter(kernel_sin)
            self.kernel_cos = nn.Parameter(kernel_cos)
            self.window_mask = nn.Parameter(window_mask)
        else:
            self.register_buffer('kernel_sin', kernel_sin)
            self.register_buffer('kernel_cos', kernel_cos)
            self.register_buffer('window_mask', window_mask)

    def get_window_mask(self, window: str, win_length: int) -> torch.Tensor:
        window_mask = scipy.signal.get_window(window, int(win_length), fftbins=True)
        window_mask = pad_center(window_mask, self.n_fft, axis=-1, mode="constant", constant_values=0)
        return torch.tensor(window_mask).to(dtype=torch.float32)

    def create_fourier_kernels(self) -> (torch.Tensor, torch.Tensor):
        freqs = numpy.arange(self.n_fft // 2 + 1)
        phases = (2 * numpy.pi) * numpy.linspace(0, 1, self.n_fft, endpoint=False)
        kernel_sin = numpy.sin(numpy.outer(freqs, phases))
        kernel_cos = numpy.cos(numpy.outer(freqs, phases))
        return (
            torch.tensor(kernel_sin).to(dtype=torch.float32),
            torch.tensor(kernel_cos).to(dtype=torch.float32),
        )

    @property
    def w_sin(self):
        return (self.kernel_sin * self.window_mask)[:, None, :]

    @property
    def w_cos(self):
        return (self.kernel_cos * self.window_mask)[:, None, :]

    @property
    def sin_inv(self):
        return torch.cat((self.kernel_sin, -self.kernel_sin[1:-1].flip(0)), 0) * self.window_mask[:, None]

    @property
    def cos_inv(self):
        return torch.cat((self.kernel_cos, self.kernel_cos[1:-1].flip(0)), 0) * self.window_mask[:, None]


class Base(torch.nn.Module, abc.ABC):
    def __init__(self, n_fft: int = 2048, win_length: int = None, hop_length: int = None, window='hann',
                 pad_mode="reflect", trainable=True, stft_format='real_imag', input_length: int = None,
                 log_mode=False):
        """
        Args:
            n_fft: Number of DFT points
            win_length: Window length. Defaults to n_fft
            hop_length: Hop length. Defaults to n_fft // 4
            window: Window function. Defaults to Hann window
            pad_mode: Padding mode. Defaults to "reflect"
            trainable: Whether to train the STFT kernels. Defaults to True
            stft_format: Output format of STFT. Defaults to "Complex"
            input_length: Whether the input length is fixed. Defaults to False (set for speed up istft)
        """
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length or self.n_fft
        self.window_name = window
        self.stride = hop_length or self.n_fft // 4

        assert pad_mode in ['constant', 'reflect', None]
        self.pad_mode = pad_mode
        assert stft_format in ['real_imag', 'mag_phase']
        self.stft_format = stft_format
        assert ~(trainable and log_mode), "Should not set trainable=True and log_mode=True at the same time"
        if log_mode:
            self.log_encoder = lambda x: torch.log(x + EPS)
            self.log_decoder = lambda x: torch.exp(x)
        else:
            self.log_encoder = lambda x: x
            self.log_decoder = lambda x: x

        self.trainable = trainable

        self.pad_amount = self.n_fft // 2
        self.input_length = input_length
        self.recalculate_window_mask = (self.input_length is None) or self.trainable

    def init_kernel(self):
        return Kernel(self.n_fft, self.window_name, self.win_length, self.trainable)

    def format_output(self, spec_real, spec_imag):
        """
        Args:
            spec_real: (batch, freqs_num, time_steps)
            spec_imag: (batch, freqs_num, time_steps)
        """
        if self.stft_format == "real_imag":
            return torch.stack((spec_real, spec_imag), 1)
        else:
            phase = torch.atan2(spec_imag, spec_real)
            mag_2 = spec_real.pow(2) + spec_imag.pow(2)
            mag = torch.sqrt(mag_2)
            mag = self.log_encoder(mag)
            return torch.stack((mag, phase), 1)

    def format_input(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Inverse function of format_output
        """
        if self.stft_format == "real_imag":
            spec_real = x[:, 0]
            spec_imag = x[:, 1]
            return spec_real, spec_imag
        else:
            mag, phase = x[:, 0], x[:, 1]
            mag = self.log_decoder(mag)
            spec_real = mag * torch.cos(phase)
            spec_imag = mag * torch.sin(phase)
            return spec_real, spec_imag

    @staticmethod
    def plot_kernel(data2plot: numpy.ndarray, ax2p, title: str):
        im = ax2p.imshow(data2plot, cmap='rainbow', aspect='auto', interpolation='nearest')
        ax2p.figure.colorbar(im, ax=ax2p)
        ax2p.set_title(title)


class STFT(Base):
    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self.padding = self.init_padding()

        kernel = self.init_kernel()
        if self.trainable:
            self.w_sin = torch.nn.Parameter(kernel.w_sin.data)
            self.w_cos = torch.nn.Parameter(kernel.w_cos.data)
        else:
            self.register_buffer('w_sin', kernel.w_sin.data)
            self.register_buffer('w_cos', kernel.w_cos.data)

    def stft(self, x: torch.Tensor):
        """
        Args:
            x: (batch, audio_length) dtype: torch.float
        Returns:
            (batch, 2, freqs_num, time_steps) dtype: torch.float
            or spec(batch, freqs_num, time_steps), phase(batch, freqs_num, time_steps)
        """
        x = x.unsqueeze(1)  # fits into a Conv1d
        x = self.padding(x)

        # Doing STFT by using torch.conv1d
        spec_imag = torch.conv1d(x, self.w_sin, stride=self.stride)
        spec_real = torch.conv1d(x, self.w_cos, stride=self.stride)

        out = self.format_output(spec_real, -spec_imag)  # Remember the minus sign for imaginary part

        return out

    def forward(self, x):
        return self.stft(x)

    def init_padding(self):
        if self.pad_mode == "constant":
            return nn.ConstantPad1d(self.pad_amount, 0)
        elif self.pad_mode == "reflect":
            return nn.ReflectionPad1d(self.pad_amount)
        else:
            self.pad_amount = 0
            return nn.Identity()

    def plot_stft_kernel(self, ax2p_sin=None, ax2p_cos=None, title='stft_kernel'):
        """
        Plot the STFT kernel
        Example:
                >>> stft_model = STFT()
                >>> from matplotlib import pyplot as plt
                >>> fig, ax2plot = plt.subplots(2, 1, layout='constrained')
                >>> stft_model.plot_kernel(ax2plot[0], ax2plot[1])
                >>> fig.show()
        """
        if ax2p_sin:
            self.plot_kernel(t2n(self.w_sin.squeeze()), ax2p_sin, f"{title}_sin")
        if ax2p_cos:
            self.plot_kernel(t2n(self.w_cos.squeeze()), ax2p_cos, f"{title}_cos")


class ISTFT(Base):
    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        kernel = self.init_kernel()
        if self.trainable:
            self.sin_inv = torch.nn.Parameter(kernel.sin_inv.data)
            self.cos_inv = torch.nn.Parameter(kernel.cos_inv.data)
        else:
            self.register_buffer('sin_inv', kernel.sin_inv.data)
            self.register_buffer('cos_inv', kernel.cos_inv.data)

        if self.recalculate_window_mask:
            self.register_buffer('window_mask', kernel.window_mask.data)
        else:
            window_sum, self.non_zero_indices = self.init_window_mask_sum(kernel.window_mask)
            self.register_buffer('window_sum', window_sum)

    def istft(self, x, expected_length=None):
        """
        Inverse function of stft
        """
        spec_real, spec_imag = self.format_input(x)
        spec_real, spec_imag = self.extend_fs(spec_real, spec_imag)

        a1 = torch.nn.functional.linear(spec_real.permute(0, 2, 1), self.cos_inv)
        b2 = torch.nn.functional.linear(spec_imag.permute(0, 2, 1), self.sin_inv)

        real_data = (a1 - b2) / self.n_fft  # Normalize the amplitude with n_fft

        real = self.overlap_add_with_window(real_data.permute(0, 2, 1), self.stride)

        expected_length = expected_length or real.shape[-1] - self.pad_amount * 2
        real = real[..., self.pad_amount: self.pad_amount + expected_length]

        return real

    def forward(self, x, expected_length=None):
        return self.istft(x, expected_length)

    @staticmethod
    def extend_fs(spec_real, spec_imag):
        """Extending the number of frequency bins from `n_fft//2+1` back to `n_fft` by
        reversing all bins except DC and Nyquist and append it on top of existing spectrogram"""
        spec_real_upper = spec_real[:, 1:-1].flip(1)
        spec_imag_upper = -spec_imag[:, 1:-1].flip(1)  # For the imaginary part, it is an odd function
        return (torch.cat((spec_real, spec_real_upper), 1),
                torch.cat((spec_imag, spec_imag_upper), 1))

    def init_window_mask_sum(self, window_mask):
        f_len = (self.input_length + 2 * self.pad_amount - self.n_fft) // self.stride + 1
        window_mask = window_mask ** 2
        window_mask = window_mask[None, :, None].repeat(1, 1, f_len)
        window_sum = overlap_add(window_mask, self.stride)[0]
        non_zero_indices = window_sum != 0  # use !=0 instead of +tiny to avoid modulation effects
        return window_sum, non_zero_indices

    def overlap_add_with_window(self, x: torch.Tensor, stride):
        if self.recalculate_window_mask:
            window = self.window_mask ** 2
            window = window[None, :, None].repeat(1, 1, x.shape[-1])
            fold_in = torch.cat([x, window], dim=0)
            fold_out = overlap_add(fold_in, stride)
            y, window_sum = fold_out[:-1], fold_out[-1]
            none_zero_indices = window_sum != 0.  # use !=0 instead of +tiny to avoid modulation effects
        else:
            y = overlap_add(x, stride)
            window_sum = self.window_sum
            none_zero_indices = self.non_zero_indices
        y[:, none_zero_indices] /= window_sum[none_zero_indices]
        return y

    def plot_istft_kernel(self, ax2p_sin=None, ax2p_cos=None, title='istft_kernel'):
        """
        Plot the ISTFT kernel
        Example:
                >>> stft_model = STFT()
                >>> from matplotlib import pyplot as plt
                >>> fig, ax2plot = plt.subplots(2, 1, layout='constrained')
                >>> stft_model.plot_kernel(ax2plot[0], ax2plot[1])
                >>> fig.show()
        """
        if ax2p_sin:
            self.plot_kernel(t2n(self.sin_inv), ax2p_sin, f"{title}_sin")
        if ax2p_cos:
            self.plot_kernel(t2n(self.cos_inv), ax2p_cos, f"{title}_cos")


class Model(STFT, ISTFT):
    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        assert self.trainable, 'Please use the STFT and ISTFT classes for un-trainable models'
        self.padding = self.init_padding()
        self.kernel = self.init_kernel()

    @property
    def w_sin(self) -> torch.Tensor:
        return self.kernel.w_sin

    @property
    def w_cos(self) -> torch.Tensor:
        return self.kernel.w_cos

    @property
    def sin_inv(self) -> torch.Tensor:
        return self.kernel.sin_inv

    @property
    def cos_inv(self) -> torch.Tensor:
        return self.kernel.cos_inv

    @property
    def window_mask(self) -> torch.Tensor:
        return self.kernel.window_mask

    def forward(self, x):
        raise RuntimeError("Please specify whether you want to use the stft or istft function")
