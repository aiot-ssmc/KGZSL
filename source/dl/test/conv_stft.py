import numpy
import torch

import libs.audio
import dl
import utils
import pytest

import logging


@pytest.mark.parametrize("trainable, fix_length", [(True, True), (True, False), (False, True), (False, False)])
def test_conv_stft(trainable, fix_length):
    audio_data = libs.audio.load(utils.file.DATA_PATH.joinpath("temp/sample.wav"))
    audio_data = audio_data[:2048 * 50]
    # utils.plot.audio_spectrogram(audio_data)
    data_type = torch.float32

    def test_func(stft_func, istft_func):
        batch_input = torch.tensor(audio_data, dtype=data_type)[None, :].repeat((3, 1))
        y = stft_func(batch_input)

        magnitude = numpy.log(y[0][0].detach().numpy() + 1e-9) * 10
        # utils.plot.spectrogram(spectrogram_data=magnitude, title="STFT Magnitude")
        x = istft_func(y).detach().numpy()
        # utils.plot.audio_spectrogram(x[0])
        diff = numpy.abs(x - audio_data)
        assert (diff < 1e-6).all()
        logging.info(f"diff : {(diff > 1e-6).nonzero()}")
        logging.info(f"diff max: {diff.max()}")

    from dl.model import conv_stft
    data_len = len(audio_data) if fix_length else None
    model_args = {"stft_format": "mag_phase", "trainable": trainable, "input_length": data_len}
    if trainable:
        model = conv_stft.Model(**model_args).to(dtype=data_type)
        dl.print_all_parameters(model, network_name='model', out_func=logging.info)
        test_func(model.stft, model.istft)
    stft_model = conv_stft.STFT(**model_args).to(dtype=data_type)
    dl.print_all_parameters(stft_model, network_name='stft_model', out_func=logging.info)
    istft_model = conv_stft.ISTFT(**model_args).to(dtype=data_type)
    dl.print_all_parameters(istft_model, network_name='istft_model', out_func=logging.info)
    test_func(stft_model, istft_model)

    # from matplotlib import pyplot as plt
    # fig, ax2plot = plt.subplots(2, 2, layout='constrained')
    # fig.set_size_inches(16, 10)
    # stft_model.plot_stft_kernel(ax2plot[0, 0], ax2plot[0, 1])
    # istft_model.plot_istft_kernel(ax2plot[1, 0], ax2plot[1, 1])
    # fig.show()

    logging.info(f"{trainable=},{fix_length=} test_conv_stft passed.")
