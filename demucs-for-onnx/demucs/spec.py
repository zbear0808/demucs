# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Conveniance wrapper to perform STFT and iSTFT"""

import torch as th
from demucs.stftProcess import STFT_Process, STFT_TYPE, ISTFT_TYPE, WINDOW_FUNCTIONS


def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    print("x shape", x.shape)
    custom_stft = STFT_Process(
        model_type='stft_B', 
        n_fft=n_fft, 
        hop_len=hop_length or n_fft // 4, 
        max_frames=0,  # Not important here.
        window_type='hann'
        ).eval()
    # Assuming x has shape (2, length) for stereo audio
    # Process left and right channels separately
    left_real, left_imag = custom_stft(x[0,:].reshape(1,1,length), 'reflect')
    right_real, right_imag = custom_stft(x[1,:].reshape(1,1,length), 'reflect')

    # custom_stft likely returns shape (1, 1, freqs, frames)
    # Squeeze the first two dimensions (batch, channel)
    left_real = left_real.squeeze(0).squeeze(0)    # Shape: (freqs, frames)
    left_imag = left_imag.squeeze(0).squeeze(0)    # Shape: (freqs, frames)
    right_real = right_real.squeeze(0).squeeze(0)  # Shape: (freqs, frames)
    right_imag = right_imag.squeeze(0).squeeze(0)  # Shape: (freqs, frames)

    # Stack real and imaginary parts for each channel along the last dimension
    # Shape becomes (freqs, frames, 2)
    left_complex = th.stack((left_real, left_imag), dim=-1)
    right_complex = th.stack((right_real, right_imag), dim=-1)

    # Stack left and right channels along the first dimension
    # Shape becomes (2, freqs, frames, 2)
    output_tensor = th.stack((left_complex, right_complex), dim=0)

    print("output onnx stft realimag shape", output_tensor.shape)


    # Ensure the output shape matches the original input's "other" dimensions if any
    # For a typical stereo input (2, length), other should be empty.
    # If input was (batch, 2, length), other would be (batch,)
    # output_tensor shape is already (2, freqs, frames, 2)
    # If needed, we could reshape based on *other, but let's assume simple stereo for now.
    # Example: return output_tensor.view(*other, 2, freqs, frames, 2) if needed

    # return output_tensor



    # z = th.stft(x,
    #             n_fft * (1 + pad),
    #             hop_length or n_fft // 4,
    #             window=th.hann_window(n_fft).to(x),
    #             win_length=n_fft,
    #             normalized=True,
    #             center=True,
    #             return_complex=True,
    #             pad_mode='reflect')
    # print("torch stft complex shape", z.shape)
    # _, freqs, frame = z.shape
    # return z.view(*other, freqs, frame) 
    channels, freqs, frame, realimag = output_tensor.shape
    print("output_tensor shape", output_tensor.shape)
    return output_tensor.reshape(1, channels, freqs, frame, 2).permute(0, 1, 4, 2, 3) 


def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    x = th.istft(z,
                 n_fft,
                 hop_length,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape
    return x.view(*other, length)
