from glob import glob
from tqdm import tqdm
import re
import os

import soundfile as sf
import pyworld as pw
import librosa

import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from utils.utils import makedirs

""" Speaker Embedder """
import model.speakerEmbedder as spkEmbedder
from model.speakerEmbedder.SpeakerNet import SpeakerNet


import yaml
config_ESD = yaml.load(
    open("./config/ESD_dataset.yaml", "r"), Loader=yaml.FullLoader
)

config_EmovDB = yaml.load(
    open("./config/EmovDB_dataset.yaml", "r"), Loader=yaml.FullLoader
)

ROOT = config_ESD["Paths"]["root_dataset"]

pth_ESD = ROOT + config_ESD["Paths"]["path_dataset"]
pth_save_ESD = ROOT + config_ESD["Paths"]["path_save"]

pth_EmovDB = ROOT + config_EmovDB["Paths"]["path_dataset"]
pth_save_EmovDB = ROOT + config_EmovDB["Paths"]["path_save"]

top_db = config_ESD["Preprocess"]["top_db"]


def ESD_str_to_info(pth_wav):
    """
    #=== INPUT
    * pth_wav           || str
        example. 'DATAPATH/0018/Surprise/evaluation/0018_001417.wav'
    
    #=== OUTPUT
    * Speaker Number    || int
    * Emotion State     || str
    * Dataset Mode      || str
        'train', 'eval', or 'test
    * filename          || str
    """
    
    spk_num, emo_state, data_mode, filename = pth_wav.split('/')[-4:]
    
    if data_mode == "evaluation":
        data_mode = 'eval'
        
    return spk_num, emo_state, data_mode, filename
    

def EmovDB_str_to_info(pth_wav):
    """
    #=== INPUT
    * pth_wav           || str
        example. "DIRNAME/bea/amused_1-15_0010.wav"
    
    #=== OUTPUT
    * Speaker Name      || str
    * Emotion State     || str
    * Dataset Mode      || str
        'train', 'eval', or 'test'
    * file Number       || int
    """
    
    spk_name = pth_wav.split('/')[-2]
    emo_state = pth_wav.split('/')[-1].split('_')[0]
    file_number = int(pth_wav.split('/')[-1].split('_')[-1].replace('.wav', ''))
    
    if 1 <= file_number and 30 >= file_number:
        data_mode = 'test'
    elif 31 <= file_number and 50 >= file_number:
        data_mode = 'eval'
    else:
        data_mode = 'train'
        
    if emo_state == "amused":
        emo_state = "Amused"
    elif emo_state == "anger":
        emo_state = "Angry"
    elif emo_state == "neutral":
        emo_state = "Neutral"
    elif emo_state == "sleepiness":
        emo_state = "Sleepy"
    elif emo_state == "sleepiness":
        emo_state = "Sleepy"
    elif emo_state == "disgust":
        emo_state = "Disgusted"
    elif emo_state == "Disgust":
        emo_state = "Disgusted"
        
    return spk_name, emo_state, data_mode, file_number


""" Load speaker embedder """
# before training, we get speaker embedding vector from the pre-trained speaker embedding model
# where the generalized speaker vector outputs. We use the ResNetSE34V2 from ClovaAI.
def get_speaker_embedder():    
    dim_spk = spkEmbedder.kwarg_SPEAKER['nOut']
    
    m_info = torch.load(
        "./model/speakerEmbedder/baseline_v2_smproto.model", 
        map_location=device
    )                                                                   #=== Load pretrained speaker embedder model
    
    spk_model = SpeakerNet(**spkEmbedder.kwarg_SPEAKER).to(device)      #=== SpeakerNet     
    spk_model.eval()
    
    spk_model.load_state_dict(m_info, strict=False)                     #=== load state dict
    print("SpeakerNet is ready.")
    
    return spk_model

    
    
""" ESD Preprocessing Method """    
def ESD_preprocess(pretrainedSpkModel):
    """
    #=== OUTPUT
    * trimmed wave      || numpy
        top_db is set by 20.
    """
        
    #=== Aggregate all paths of the wave files.
    _pth_glob = "{}/**/**/**/*".format(pth_ESD) # example. "DIRNAME/0018/Surprise/evaluation/0018_001417.wav"
    pth_wave_list = glob(_pth_glob) 
    print(pth_wave_list)
    pth_wave_list = [pth for pth in pth_wave_list if int(re.sub('^0.', "", pth.split("/")[-4])) > 10]
    
    pbar = tqdm(pth_wave_list)
    for pth in pbar:
        
        #=== Get Data
        wav_npy = sf.read(pth)[0]
        
        _, emo_state, data_mode, file_name = ESD_str_to_info(pth.replace(pth_ESD, ""))
        file_name = file_name.replace('.wav', '.npy')
        pbar.set_postfix({'Path' : file_name})
        
        #=== trim waveform
        wav_npy = librosa.effects.trim(wav_npy, top_db=20)[0]
        wav_npy = wav_npy / np.max(wav_npy, 1e-5) * 0.99
        wav_torch = torch.tensor(wav_npy).unsqueeze(0).float().to(device)
        
        #=== get speaker embedding
        with torch.no_grad():
            spk_emb = pretrainedSpkModel(wav_torch)     # (1, dim_spk = 512)
        spk_emb_npy = spk_emb.to('cpu').detach().numpy()
        
        #=== Save trimmed waveform and speaker embedding
        _dir_wav = os.path.join(pth_save_ESD, 'wav', str(emo_state), data_mode)
        _dir_spkEmb = os.path.join(pth_save_ESD, 'spkEmb', str(emo_state), data_mode)
        
        if not os.path.exists(_dir_wav):
            makedirs(_dir_wav)      # If there's no dir, then make it!
        if not os.path.exists(_dir_spkEmb):
            makedirs(_dir_spkEmb)      # If there's no dir, then make it!
            
        np.save(os.path.join(_dir_wav, file_name), wav_npy)             # (1, wav_length)
        np.save(os.path.join(_dir_spkEmb, file_name), spk_emb_npy)      # (1, dim_spk)
        


""" ESD Preprocessing Method """    
def EmovDB_preprocess(pretrainedSpkModel):
    """
    #=== OUTPUT
    * trimmed wave      || numpy
        top_db is set by 20.
    """
        
    #=== Aggregate all paths of the wave files.
    _pth_glob = "{}/**/*".format(pth_EmovDB) # example. "DIRNAME/bea/amused_1-15_0010.wav"
    pth_wave_list = glob(_pth_glob) 
    print(pth_wave_list)
    
    pbar = tqdm(pth_wave_list)
    for pth in pbar:
        
        #=== Get Data
        wav_npy = sf.read(pth)[0]
        
        spk_name, emo_state, data_mode, file_number = EmovDB_str_to_info(pth.replace(pth_ESD, ""))
        file_name = spk_name + "_" + "{}".format(str(file_number).zfill(4)) + ".npy"
        pbar.set_postfix({'Path' : file_name})
        
        #=== trim waveform
        wav_npy = librosa.effects.trim(wav_npy, top_db=20)[0]
        wav_npy = wav_npy / np.max(wav_npy, 1e-5) * 0.99
        wav_torch = torch.tensor(wav_npy).unsqueeze(0).float().to(device)
        
        #=== get speaker embedding
        with torch.no_grad():
            spk_emb = pretrainedSpkModel(wav_torch)     # (1, dim_spk = 512)
        spk_emb_npy = spk_emb.to('cpu').detach().numpy()
        
        #=== Save trimmed waveform and speaker embedding
        _dir_wav = os.path.join(pth_save_EmovDB, 'wav', str(emo_state), data_mode)
        _dir_spkEmb = os.path.join(pth_save_EmovDB, 'spkEmb', str(emo_state), data_mode)
        
        if not os.path.exists(_dir_wav):
            makedirs(_dir_wav)      # If there's no dir, then make it!
        if not os.path.exists(_dir_spkEmb):
            makedirs(_dir_spkEmb)      # If there's no dir, then make it!
            
        np.save(os.path.join(_dir_wav, file_name), wav_npy)             # (1, wav_length)
        np.save(os.path.join(_dir_spkEmb, file_name), spk_emb_npy)      # (1, dim_spk)
        


if __name__ == "__main__":
    pretrainedModel = get_speaker_embedder()
    pretrainedModel.eval()
    
    ESD_preprocess(pretrainedModel)
    EmovDB_preprocess(pretrainedModel)



