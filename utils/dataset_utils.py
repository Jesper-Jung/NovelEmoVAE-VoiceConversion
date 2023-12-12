import torch

from librosa.util import normalize
import librosa
import soundfile as sf
import random

import yaml
import numpy as np
import pyworld as pw

import parselmouth
from parselmouth.praat import call




def pitch_formant_argument(wav, sr=16000, 
                           min_formantShift=1, max_formantShift=1.4,
                           min_pitchShift=1, max_pitchShift=1.2,
                           min_pitchRange=1., max_pitchRange=1.15, 
                           ):
    """
    * wav       || (n_channel, t_samples), numpy
    """
    
    return_torch = type(wav) == torch.Tensor
    
    if return_torch:
        device = wav.device
        wav = wav.numpy(force=True)
        
    if len(wav.shape) == 1:
        wav = np.expand_dims(wav, axis=0)
    
    
    decision = random.random()
    
    r_fShift = min_formantShift + (max_formantShift - min_formantShift) * random.random()
    r_pShift = min_pitchShift + (max_pitchShift - min_pitchShift) * random.random()
    r_pRange = min_pitchRange + (max_pitchRange - min_pitchRange) * random.random()


    sound = parselmouth.Sound(wav, sampling_frequency=16000)
    
    ###==== Augment
    if decision >= 0.5:
        converted_sound = call(sound, "Change speaker...", 75, 600, r_fShift, r_pShift, r_pRange, 1)
    else:
        converted_sound = sound
        
    if return_torch:
        converted_wav = torch.tensor(converted_sound.values).float().to(device)
    else:
        converted_wav = np.squeeze(converted_sound.values, axis=0)
    
    return converted_wav


def get_mel_from_audio(config, audio, trim_on=True, audio_norm=True):

    """ Configure """

    n_fft = config["Preprocess"]["n_fft"]
    n_hop = config["Preprocess"]["n_hop"]
    n_mel = config["Preprocess"]["n_mel"]

    sr = config["Preprocess"]["sr"]



    """ Preprocess """

    audio = audio.astype(np.float64)

    if audio_norm:
        audio = normalize(audio)

    # audio trim
    if trim_on:
        audio, _ = librosa.effects.trim(audio, top_db=20)



    """ Get MelSpec. """

    # transform to mels
    linear = np.abs(librosa.stft(y = audio, n_fft = n_fft, hop_length = n_hop, center = False))
    
    filter_banks = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mel, fmin=0, fmax=None) # 
    mel = np.dot(filter_banks, linear) # (n_mel: 80, nb_frame)

    # normalization
    mel = normalization_compression(mel)


    return mel


def get_mel_pitch_energy(config, wav_path, trim_on=True, audio_norm=True):
    
    """ Configure """
    n_fft = config["Preprocess"]["n_fft"]
    n_hop = config["Preprocess"]["n_hop"]
    n_mel = config["Preprocess"]["n_mel"]

    sr = config["Preprocess"]["sr"]

    # load .wav file
    audio, _ = sf.read(wav_path, samplerate=None)
    audio = audio.astype(np.float64)

    if audio_norm:
        audio = normalize(audio)

    # audio trim
    if trim_on:
        audio, _ = librosa.effects.trim(audio, top_db=20)



    #=== Get Mel Spectrograms

    # transform to mels
    linear = np.abs(librosa.stft(y = audio, n_fft = n_fft, hop_length = n_hop, center = False))
    
    # ! 참고
    # librosa.filters.mel 할 때 fmin=50, fmax=7600 정도로 하는 게 국룰이다.
    filter_banks = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mel, fmin=0, fmax=None) # 
    mel = np.dot(filter_banks, linear) # (n_mel: 80, nb_frame)

    # normalization
    mel = normalization_compression(mel)
    # ! 참고 1
    # 음성 합성 쪽에서 쓰는 normalization technique가 따로 있다.
    # 아마도 ref dB 같음.



    #=== Get Pitch,
    # Extract Pitch/f0 from raw waveform using PyWORLD

    """
    f0_floor : float
        Lower F0 limit in Hz.
        Default: 71.0
    f0_ceil : float
        Upper F0 limit in Hz.
        Default: 800.0
    """

    _f0, timeaxis = pw.dio(
        audio, sr, frame_period=n_hop / sr * 1000, f0_floor=60., f0_ceil=800.
    )  # For hop size 256 frame period is 11.6 ms        # ex. (Number of Frames) = (654,)
    f0 = pw.stonemask(audio, _f0, timeaxis, fs=sr)
    f0 = f0[:mel.shape[1]]


    #=== get energy
    energy = np.sqrt(np.sum(linear ** 2, axis=0)).squeeze()


    return mel, f0, energy, audio



    """ Normalization technique
    """

def normalization_compression(mel, C = 1, min = 1e-5, exp_mode=False):
    if exp_mode:
        return np.log(np.clip(mel, a_min = min, a_max = None) * C)
    else:
        return np.log10(np.clip(mel, a_min = min, a_max = None) * C)

def normalization_AutoVC(mel, min_level = np.exp(-100 / 20 * np.log(10))):
    mel_db = 20 * np.log10(np.maximum(min_level, mel)) - 10
    S = np.clip((mel_db + 100)/100, 0, 1)
    return S

def denormalization_AutoVC(mel):
    mel = mel * 100 - 100
    mel = (10 + mel) / 20
    return mel


def crop_data(mel, pitch, energy, crop_len):
    """
    ? INPUT
    - mel: (mel_channel, mel_length)

    ? OUTPUT
    - mel, (mel_channel, cropped_mel)
    """
    assert mel.shape[1] == pitch.shape[0] == energy.shape[0], "Lengths are not consistence."
    _, mel_len = mel.shape

    if crop_len >= mel_len:
        return _pad_T(mel, crop_len), _pad_1d(pitch, crop_len), _pad_1d(energy, crop_len)
    else:
        _pad_start = np.random.randint(mel.shape[1] - crop_len)
        _pad_end = _pad_start + crop_len
        return mel[:, _pad_start : _pad_end], pitch[_pad_start : _pad_end], energy[_pad_start : _pad_end]

def crop_audio(audio, crop_len=26000):
    if crop_len > audio.shape[0]:
        return _pad_1d(audio, crop_len)
    else:
        _pad_start = np.random.randint(audio.shape[0] - crop_len)
        _pad_end = _pad_start + crop_len
        return audio[_pad_start : _pad_end]
    
def crop_mel(mel, crop_len):
    """ Cropping mel to given length.
    If mel.length < crop_len, then pad zero for all.
    Else, cropping anywhere to be selected randomly.

    Args:
        mel (numpy): (mel_channel, mel_length)
        crop_len (int): return to given length.
    """
    
    mel_channel, mel_length = mel.shape
    
    if crop_len > mel_length:
        return _pad_T(mel, crop_len)
    else:
        _pad_start = np.random.randint(mel_length - crop_len)
        _pad_end = _pad_start + crop_len
        return mel[:, _pad_start:_pad_end]
    

def _pad_T(mel, crop_len):
    pad_len = crop_len - mel.shape[1]
    return np.pad(mel, ((0, 0), (0, pad_len)))

def _pad_1d(data, crop_len):
    pad_len = crop_len - data.shape[0]
    return np.pad(data, (0, pad_len))




""" For Computing Statistics """

def is_outlier(x, p25, p75):
    """Check if value is an outlier."""
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    #
    return x <= lower or x >= upper


def remove_outlier(x):
    """Remove outlier from x."""
    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)
    zero_idxs = np.where(x == 0.0)[0]
    indices_of_outliers = []
    for ind, value in enumerate(x):
        if is_outlier(value, p25, p75):
            indices_of_outliers.append(ind)
    #
    x[indices_of_outliers] = 0.0
    #
    # replace by mean f0.
    x[indices_of_outliers] = np.max(x)
    x[zero_idxs] = 0.0
    return x