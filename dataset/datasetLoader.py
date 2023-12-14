from glob import glob
import os
import sys
sys.path.append('./dataset')
import yaml
import re
import time

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import random

prev_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(prev_dir)

from utils.dataset_utils import crop_mel
from utils.dataset_utils import pitch_formant_argument
import dataset

""" ESD Dataset Configuration """
config_ESD = yaml.load(
    open("./config/ESD_dataset.yaml", "r"), Loader=yaml.FullLoader
)

config_EmovDB = yaml.load(
    open("./config/EmovDB_dataset.yaml", "r"), Loader=yaml.FullLoader
)

""" Build ESD dataset class """

class Dataset_ESD(Dataset):
    def __init__(self, config, dataset_mode='train'):
        """  Dataset 
        
        mel_cropLength: length of mel feature.
        dataset_mode: 'train', 'eval', or 'test'
        
        """
        self.sr = config['Dataset_Info']['sr']
        length_mel = config['Preprocess']['length_mel']
        n_fft = config['Preprocess']['n_fft']
        hop_size = config['Preprocess']['hop_size']
        
        length_wav = length_mel * hop_size
        
        # example. 'DATAPATH/wav/Happy/eval/0020_000716.npy'
        data_path = config['Paths']['root_dataset'] + config['Paths']['path_save']
        self.wav_list = sorted(glob("{}/wav/**/{}/**.npy".format(data_path, dataset_mode)))
        self.cont_list = sorted(glob("{}/cont/**/{}/**.npy".format(data_path, dataset_mode)))
        self.disc_list = sorted(glob("{}/disc/**/{}/**.npy".format(data_path, dataset_mode)))
        
        self.mode = dataset_mode
        self.crop_len = length_wav
        
        # Emotion state codebook for ESD dataset.
        self.Codebook_EMO = {
            0: "neutral", 1: "angry", 2: "happy", 3: "sad", 4: "excited"
        }

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        """ Total 15000 Train Set, 1000 Eval Set, and 1500 Test Set.
        In the train set (ESD), there are 1500 set of each of speakers, and 3000 set of each of emotion.

        #=== Contents
        - mel: mel-spectrogram, cropped by setting length
        - speaker_id
        - emotion_id: [neutral, Angry, Happy, Sad, Surprise] <-> [0, 1, 2, 3, 4]
        """

        ###==== arguments 
        """
            random pitch, random formant, crop, random gain
        """
        
        wav, disc, cont = np.load(self.wav_list[idx]), np.load(self.disc_list[idx]), np.load(self.cont_list[idx])
        wav, disc, cont = self.crop_and_gain(wav, disc, cont, self.crop_len) # (cropped_len)

        ### Speaker ID
        spk_emb = np.load(self.wav_list[idx].replace('wav', 'spkEmb'))
        if len(spk_emb.shape) == 2:
            spk_emb = np.squeeze(spk_emb, axis=0)

        ### Emotion State
        
        emo_state = self.wav_list[idx].split('/')[-3]       # Angry, Happy, Neutral, Sad, Surpirse
        
        return {
            'wav': wav,
            'disc': disc,
            'cont': cont,
            'spk_emb': spk_emb,
            'emo_state': emo_state,
            'pth': self.wav_list[idx]
        }
        
        
       
    def crop_and_gain(self, wav, discUnit, contUnit, crop_len):
        if len(wav.shape) == 2:
            wav = wav.squeeze(0)
            
        if len(discUnit.shape) == 3:
            discUnit = discUnit.squeeze(0)
            
        if len(contUnit.shape) == 3:
            contUnit = contUnit.squeeze(0)
            
        #=== Crop (wav)
        if crop_len >= wav.shape[0]:
            wav_cropped = self._pad_1d(wav, crop_len)
            contUnit_cropped = self._pad_2d_T(contUnit, crop_len // 320)
            discUnit_cropped = self._pad_1d(discUnit, crop_len // 320)
        else:                                           # Hubert -> 320 hop size.
            k = wav.shape[0]//320 - crop_len//320
            _pad_frame_start = np.random.randint(wav.shape[0]//320 - crop_len//320) if k > 0 else 0
            _pad_frame_end = _pad_frame_start + crop_len // 320
            
            wav_cropped = wav[_pad_frame_start * 320 : _pad_frame_end * 320]
            contUnit_cropped = contUnit[_pad_frame_start : _pad_frame_end]
            discUnit_cropped = discUnit[_pad_frame_start : _pad_frame_end]
            
            
        #=== Gain
        if self.mode == "train":
            gain = random.random() * (0.99 - 0.4) + 0.4
            flip = -1 if random.random() > 0.5 else 1
            wav_cropped = flip * gain * wav_cropped / max(np.max(np.abs(wav_cropped)), 1e-5)
            
        
            
        return wav_cropped, discUnit_cropped, contUnit_cropped
        
     
    def _pad_1d(self, data, crop_len):
        pad_len = crop_len - data.shape[0]
        return np.pad(data, (0, pad_len))
    
    def _pad_2d_T(self, data, crop_frame_len):
        pad_len = crop_frame_len - data.shape[0]
        padded = np.pad(data, ((0, pad_len), (0, 0)))
        return padded
        
        

""" Build ESD dataset class """

class Dataset_EmovDB(Dataset):
    def __init__(self, config, dataset_mode='train'):
        """  Dataset 
        
        mel_cropLength: length of mel feature.
        dataset_mode: 'train', 'eval', or 'test'
        
        """
        self.sr = config['Dataset_Info']['sr']
        length_mel = config['Preprocess']['length_mel']
        n_fft = config['Preprocess']['n_fft']
        hop_size = config['Preprocess']['hop_size']
        
        length_wav = length_mel * hop_size
        
        # example. 'DATAPATH/wav/Happy/eval/0020_000716.npy'
        data_path = config['Paths']['root_dataset'] + config['Paths']['path_save']
        self.wav_list = sorted(glob("{}/wav/**/{}/**.npy".format(data_path, dataset_mode)))
        self.cont_list = sorted(glob("{}/cont/**/{}/**.npy".format(data_path, dataset_mode)))
        self.disc_list = sorted(glob("{}/disc/**/{}/**.npy".format(data_path, dataset_mode)))
        
        self.mode = dataset_mode
        self.crop_len = length_wav
        
        # Emotion state codebook for ESD dataset.
        self.Codebook_EMO = {
            0: "neutral", 1: "angry", 2: "happy", 3: "sad", 4: "excited"
        }

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        """ Total 15000 Train Set, 1000 Eval Set, and 1500 Test Set.
        In the train set (ESD), there are 1500 set of each of speakers, and 3000 set of each of emotion.

        #=== Contents
        - mel: mel-spectrogram, cropped by setting length
        - speaker_id
        - emotion_id: [neutral, Angry, Happy, Sad, Surprise] <-> [0, 1, 2, 3, 4]
        """

        ###==== arguments 
        """
            random pitch, random formant, crop, random gain
        """
        wav, disc, cont = np.load(self.wav_list[idx]), np.load(self.disc_list[idx]), np.load(self.cont_list[idx])
        wav, disc, cont = self.crop_and_gain(wav, disc, cont, self.crop_len) # (cropped_len)

        ### Speaker ID
        spk_emb = np.load(self.wav_list[idx].replace('wav', 'spkEmb'))
        if len(spk_emb.shape) == 2:
            spk_emb = np.squeeze(spk_emb, axis=0)

        ### Emotion State
        
        emo_state = self.wav_list[idx].split('/')[-3]       # Angry, Happy, Neutral, Sad, Surpirse
        
        return {
            'wav': wav,
            'disc': disc,
            'cont': cont,
            'spk_emb': spk_emb,
            'emo_state': emo_state,
            'pth': self.wav_list[idx]
        }
        
        
    def crop_and_gain(self, wav, discUnit, contUnit, crop_len):        
        if len(wav.shape) == 2:
            wav = wav.squeeze(0)
            
        if len(discUnit.shape) == 3:
            discUnit = discUnit.squeeze(0)
            
        if len(contUnit.shape) == 3:
            contUnit = contUnit.squeeze(0)
            
        #=== Crop (wav)
        if crop_len >= wav.shape[0]:
            wav_cropped = self._pad_1d(wav, crop_len)
            contUnit_cropped = self._pad_2d_T(contUnit, crop_len // 320)
            discUnit_cropped = self._pad_1d(discUnit, crop_len // 320)
        else:                                           # Hubert -> 320 hop size.
            k = wav.shape[0]//320 - crop_len//320
            _pad_frame_start = np.random.randint(wav.shape[0]//320 - crop_len//320) if k > 0 else 0
            _pad_frame_end = _pad_frame_start + crop_len // 320
            
            wav_cropped = wav[_pad_frame_start * 320 : _pad_frame_end * 320]
            contUnit_cropped = contUnit[_pad_frame_start : _pad_frame_end]
            discUnit_cropped = discUnit[_pad_frame_start : _pad_frame_end]
            
            
        #=== Gain
        if self.mode == "train":
            gain = random.random() * (0.99 - 0.4) + 0.4
            flip = -1 if random.random() > 0.5 else 1
            wav_cropped = flip * gain * wav_cropped / max(np.max(np.abs(wav_cropped)), 1e-5)
            
        return wav_cropped, discUnit_cropped, contUnit_cropped
        
     
    def _pad_1d(self, data, crop_len):
        pad_len = crop_len - data.shape[0]
        return np.pad(data, (0, pad_len))
    
    def _pad_2d_T(self, data, crop_frame_len):
        pad_len = crop_frame_len - data.shape[0]
        padded = np.pad(data, ((0, pad_len), (0, 0)))
        return padded
        
 
            
            

def collate_fn(batch):
    batch_size = len(batch)
    
    wavs = np.array([b['wav'] for b in batch], dtype=np.float64)
    conts = np.array([b['cont'] for b in batch], dtype=np.float64)
    discs = np.array([b['disc'] for b in batch], dtype=np.int64)
    spkEmbs = np.array([b['spk_emb'] for b in batch], dtype=np.float64)
    emoIDs = np.array([dataset.Codebook_EmoState[b['emo_state']] for b in batch], dtype=np.int64)
    
    wav_tensor = torch.tensor(wavs).float()
    cont_tensor = torch.tensor(conts).float()
    disc_tensor = torch.tensor(discs).long()
    spkEmb_tensor = torch.tensor(spkEmbs).float()
    emoID_tensor = torch.tensor(emoIDs).long()
    
    return wav_tensor, cont_tensor, disc_tensor, spkEmb_tensor, emoID_tensor



if __name__ == "__main__":
    datasetESD = Dataset_ESD(config_ESD)
    datasetEmovDB = Dataset_EmovDB(config_EmovDB)
    batch_size = 4
    
    loader = DataLoader(
        datasetESD, batch_size=batch_size, 
        shuffle=True, collate_fn=collate_fn
    )

    for i, data in enumerate(loader):
        # _data = F.normalize(data[2], p=2, dim=1)
        # _data = data[2]
        # c = torch.cdist(_data, _data).detach().cpu().numpy()
        #print(c)
        
        print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
        
        if i == 10:
            break