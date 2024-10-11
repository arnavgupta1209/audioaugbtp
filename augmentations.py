import pandas as pd
import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
#from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import IPython.display as ipd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models

def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_end_len = max_len - sig_len

      # Pad with 0s
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((sig, pad_end), 1)
      
    return (sig, sr)

def repeat_pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    while (sig_len != max_len):
        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        if (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_end_len = max_len - sig_len
    
            pad_end = sig[:,:pad_end_len]
    
            sig = torch.cat((sig, pad_end), 1)
        num_rows, sig_len = sig.shape

    return (sig, sr)

def time_shift(aud, shift_limit):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

def time_shift_spectro(spectro, shift_limit):
    #spectro is 3d with 1 channel dimension
    _, freq_len, time_len = spectro.shape
    shift_amt = int(shift_limit * time_len)
    return (spectro.roll(shift_amt, dims=2))

def add_noise_spectro(spectro, noise_factor):
    noise = torch.randn(spectro.shape) * noise_factor
    return spectro + noise

def specaugment(spectro, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, freq_len, time_len = spectro.shape
    freq_mask_size = int(freq_len * max_mask_pct)
    time_mask_size = int(time_len * max_mask_pct)
    min_val = spectro.min()
    
    # Frequency masking
    for _ in range(n_freq_masks):
        f = random.randint(0, freq_len - freq_mask_size)
        spectro[:, f:f+freq_mask_size, :] = min_val

    # Time masking
    for _ in range(n_time_masks):
        t = random.randint(0, time_len - time_mask_size)
        spectro[:, :, t:t+time_mask_size] = min_val

    return spectro
