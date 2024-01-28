from glob import glob
import os
import sys
sys.path.append('./dataset')
import yaml
import re
import time

from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else 'cpu'

import numpy as np
import random

prev_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(prev_dir)

from utils.dataset_utils import crop_mel
from utils.dataset_utils import pitch_formant_argument
import dataset



""" Build Dataset Class """

class EmotionSpeechDataset(Dataset):
    def __init__(self, config, config_preprocess, dataset_mode='train', synthesizer=False):
        """  Dataset 
        
        mel_cropLength: length of mel feature.
        dataset_mode: 'train', 'eval', or 'test'
        
        """
        self.sr = config['Dataset']['sr']
        self.use_pitch_shift = config['Dataset']['use_pitch_shift']
        hop_size = config['Dataset']['hop_size']
        n_fft = config['Dataset']['n_fft']
        length_mel = config['Dataset']['length_mel']
        length_wav = length_mel * hop_size
        
        
        # example. 'DATAPATH/wav/Happy/eval/0020_000716.npy'
        assert dataset_mode in ['train', 'test_s2s', 'test_u2u'], "dataset_mode needs to be 'train' or 'test'"
        self.dataset_mode = dataset_mode
        self.synthesizer = synthesizer
        
        self.list_wav = self.__get_path(config_preprocess)
        self.crop_len = length_wav
        
        # Emotion state codebook for ESD dataset.
        self.Codebook_EMO = {
            0: "neutral", 1: "angry", 2: "happy", 3: "sad", 4: "excited"
        }

    def __len__(self):
        return len(self.list_wav)

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
        
        ### wav & unit
        wav = np.load(self.list_wav[idx])
        unit = np.load(self.list_wav[idx].replace('wav', 'unit'))
        
        if not self.synthesizer:
            wav, unit = self.crop_gain(wav, unit, self.crop_len) # (cropped_len)

        
        ### Speaker Embedding
        spk_emb = np.load(self.list_wav[idx].replace('wav', 'spkEmb'))
        if len(spk_emb.shape) == 2:
            spk_emb = np.squeeze(spk_emb, axis=0)

        ### Emotion State
        emo_state = self.list_wav[idx].split('/')[-1].split('_')[0]       # Angry, Happy, Neutral, Sad, Surpirse
        
        return {
            'wav': wav,
            'unit': unit,
            'spk_emb': spk_emb,
            'emo_state': emo_state,
            'pth': self.list_wav[idx]
        }
        
        
       
    def crop_gain(self, wav, unit, crop_len):
        if len(wav.shape) == 2:
            wav = wav.squeeze(0)
            
        if len(unit.shape) == 3:
            unit = unit.squeeze(0)
            
            
        #=== Crop (wav)
        if crop_len >= wav.shape[-1]:
            wav_cropped = self.__pad_1d(wav, crop_len)
            unit_cropped = self.__pad_2d_T(unit, crop_len // 320)

        else:                                           # Hubert -> 320 hop size.
            k = wav.shape[-1]//320 - crop_len//320
            _pad_frame_start = np.random.randint(wav.shape[-1]//320 - crop_len//320) if k > 0 else 0
            _pad_frame_end = _pad_frame_start + crop_len // 320
            
            wav_cropped = wav[_pad_frame_start * 320 : _pad_frame_end * 320]
            unit_cropped = unit[_pad_frame_start : _pad_frame_end]

            
        #=== Gain
        if self.dataset_mode == "train":
            gain = random.random() * (0.99 - 0.4) + 0.4
            flip = -1 if random.random() > 0.5 else 1
            wav_cropped = flip * gain * wav_cropped / max(np.max(np.abs(wav_cropped)), 1e-5)


        return wav_cropped, unit_cropped
        
        
    
    def __get_path(self, config_preprocess):
        ROOT = config_preprocess['Root_Dataset']
        
        list_test_speakers = config_preprocess['ESD']['list_test_speakers']
        
        dict_savePath_dataset = {
            'ESD': ROOT + config_preprocess['ESD']["Paths"]["path_save"],
            'EmovDB': ROOT + config_preprocess['EmovDB']["Paths"]["path_save"],
            'RAVDESS': ROOT + config_preprocess['RAVDESS']["Paths"]["path_save"],
            'JL_Corpus': ROOT + config_preprocess['JL_Corpus']["Paths"]["path_save"],
        }
        
        #==== Get Paths 
        list_ESD_wav = sorted(glob("{}/wav/**.npy".format(dict_savePath_dataset['ESD'])))
        list_EmovDB_wav = sorted(glob("{}/wav/**.npy".format(dict_savePath_dataset['EmovDB'])))
        list_RAVDESS_wav = sorted(glob("{}/wav/**.npy".format(dict_savePath_dataset['RAVDESS'])))
        list_JL_Corpus_wav = sorted(glob("{}/wav/**.npy".format(dict_savePath_dataset['JL_Corpus'])))
        
        # ESD split
        list_ESD_test_wav = [pth for pth in list_ESD_wav if pth.split('/')[-1].split('_')[1] in list_test_speakers \
                    and (int(pth.split('/')[-1].split('_')[-1].replace('.npy', '')) - 1) % 350 >= 320]
        list_ESD_train_wav = [pth for pth in list_ESD_wav if (int(pth.split('/')[-1].split('_')[-1].replace('.npy', '')) - 1) % 350 < 320]

        if self.dataset_mode == 'test_s2s':
            list_wav = list_ESD_test_wav
        
        elif self.dataset_mode == 'test_u2u':
            list_wav = [pth for pth in list_JL_Corpus_wav if 'a_1.npy' in pth]
            
        elif self.dataset_mode == 'train':
            list_wav = list_ESD_train_wav + list_EmovDB_wav
            
        return list_wav
        
        
    
    def __pad_1d(self, data, crop_len):
        pad_len = crop_len - data.shape[-1]
        return np.pad(data, (0, pad_len))
    
    def __pad_1d_torch(self, data, crop_len):
        pad_len = crop_len - data.shape[-1]
        return F.pad(data, (0, pad_len), "constant", 0)
        
    def __pad_2d_T(self, data, crop_frame_len):
        pad_len = crop_frame_len - data.shape[0]
        padded = np.pad(data, ((0, pad_len), (0, 0)))
        return padded
        
        

   

def collate_fn(batch):
    batch_size = len(batch)
    
    wavs = np.array([b['wav'] for b in batch], dtype=np.float64)
    unit = np.array([b['unit'] for b in batch], dtype=np.float64)
    spkEmbs = np.array([b['spk_emb'] for b in batch], dtype=np.float64)
    emoIDs = np.array([dataset.Codebook_EmoState[b['emo_state']] for b in batch], dtype=np.int64)
    
    wav_tensor = torch.tensor(wavs).float()
    unit_tensor = torch.tensor(unit).float()
    spkEmb_tensor = torch.tensor(spkEmbs).float()
    emoID_tensor = torch.tensor(emoIDs).long()
    
    return wav_tensor, unit_tensor, spkEmb_tensor, emoID_tensor



if __name__ == "__main__":
    import time
    
    config = yaml.load(
        open("./config/config.yaml", "r"), Loader=yaml.FullLoader
    )
    
    config_preprocess = yaml.load(
        open("./config/config_preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    
    datasetESD = EmotionSpeechDataset(config, config_preprocess, dataset_mode='test')
    batch_size = 4
    
    from tqdm import tqdm
    
    loader = DataLoader(
        datasetESD, batch_size=batch_size, 
        shuffle=False, collate_fn=collate_fn
    )

    for i, data in tqdm(enumerate(loader)):
        # _data = F.normalize(data[2], p=2, dim=1)
        # _data = data[2]
        # c = torch.cdist(_data, _data).detach().cpu().numpy()
        #print(c)
        
        #print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
        
        if i == 10:
            break
