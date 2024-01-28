import torch
import torch.nn.functional as f

import numpy as np
import os
import sys
import yaml
from tqdm import tqdm
import soundfile as sf

sys.path.append('./dataset')
from datasetLoader import EmotionSpeechDataset

config = yaml.load(
        open("./config/config.yaml", "r"), Loader=yaml.FullLoader
)
    
config_preprocess = yaml.load(
    open("./config/config_preprocess.yaml", "r"), Loader=yaml.FullLoader
)
   
   
   
def get_all_length_of_dataset(dataset_mode):
    assert dataset_mode in ['train', 'test_s2s', 'test_u2u'], 'ㅂㅅ'
    
    datasetESD = EmotionSpeechDataset(config, config_preprocess, dataset_mode=dataset_mode)
        
    tot_length_ESD = 0.
    tot_length_EmovDB = 0.
    tot_length_JL = 0.
    dict_emo_length_ESD = {"Angry": 0., "Happy": 0., "Sad": 0., "Neutral": 0., "Surprise": 0.}
    dict_emo_length_EmovDB = {"Angry": 0., "Amused": 0., "Sleepy": 0.,"Neutral": 0.}
    dict_emo_length_JL = {"Angry": 0., "Happy": 0., "Sad": 0., "Neutral": 0., "Excited": 0.}
    
    n_ESD = 0
    n_EmovDB = 0
    n_JL = 0
    dict_emo_n_ESD = {"Angry": 0, "Happy": 0, "Sad": 0, "Neutral": 0, "Surprise": 0}
    dict_emo_n_EmovDB = {"Angry": 0, "Amused": 0, "Sleepy": 0,"Neutral": 0}
    dict_emo_n_JL = {"Angry": 0, "Happy": 0, "Sad": 0, "Neutral": 0, "Excited": 0}
    
    for data in tqdm(datasetESD):
        __pth, __emo_state = data['pth'], data['emo_state']
        
        if 'ESD' in __pth:
            __wav = np.load(__pth)
            
            tot_length_ESD += len(__wav) / 16000
            dict_emo_length_ESD[__emo_state] += len(__wav) / 16000
            
            n_ESD += 1
            dict_emo_n_ESD[__emo_state] += 1
            
        elif 'EmovDB' in __pth:
            __wav = np.load(__pth)
            
            tot_length_EmovDB += len(__wav) / 16000
            dict_emo_length_EmovDB[__emo_state] += len(__wav) / 16000
            
            n_EmovDB += 1
            dict_emo_n_EmovDB[__emo_state] += 1
            
        elif 'JL_Corpus' in __pth:
            __wav = np.load(__pth)
            
            tot_length_JL += len(__wav) / 16000
            dict_emo_length_JL[__emo_state] += len(__wav) / 16000
            
            n_JL += 1
            dict_emo_n_JL[__emo_state] += 1
            
            
    
    if n_ESD:
        print("Total length of ESD: {}".format(tot_length_ESD))
        print("Avg. utterance duration of ESD: {}".format(tot_length_ESD / n_ESD))
        print("Length of ESD for each emotions: {}".format(dict_emo_length_ESD))
        print("N of ESD for each emotions: {}".format(dict_emo_n_ESD))
    
    if n_EmovDB:
        print("Total length of EmovDB: {}".format(tot_length_EmovDB))
        print("Avg. utterance duration of EmovDB: {}".format(tot_length_EmovDB / n_EmovDB))
        print("Length of EmovDB for each emotions: {}".format(dict_emo_length_EmovDB))
        print("N of EmovDB for each emotions: {}".format(dict_emo_n_EmovDB))
        
    if n_JL:
        print("Total length of JL: {}".format(tot_length_JL))
        print("Avg. utterance duration of JL: {}".format(tot_length_JL / n_JL))
        print("Length of JL for each emotions: {}".format(dict_emo_length_JL))
        print("N of JL for each emotions: {}".format(dict_emo_n_JL))
        
    
    
    
    
    
    
if __name__ == "__main__":
    # get_all_length_of_dataset('train')
    get_all_length_of_dataset('test_s2s')
    get_all_length_of_dataset('test_u2u')

