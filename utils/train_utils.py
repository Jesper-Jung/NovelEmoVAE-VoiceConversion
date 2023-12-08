import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt
import librosa.display as display

import os


def check_recon_mel(recon_mel, pth_saveMel, number, mode='GT'):
    """
        Save reconstruction log-mel spectrogram.
    """
    fig, ax = plt.subplots()
    
    im = ax.imshow(recon_mel, aspect="auto", origin="lower", interpolation="none")
    
    ax.set(title='Mel-frequency spectrogram')    
    plt.colorbar(im, ax=ax, format='%+2.0f dB')
    
    fig.canvas.draw()
    
    if mode != "GT":
        plt.savefig(pth_saveMel + f"/{mode}_mel_{number}.jpg")
        np.save(pth_saveMel + f"/{mode}_mel_{number}.npy", recon_mel)
    else:
        plt.savefig(pth_saveMel + f"/GT_mel.jpg")
        np.save(pth_saveMel + f"/GT_mel.npy", recon_mel)

    plt.close()