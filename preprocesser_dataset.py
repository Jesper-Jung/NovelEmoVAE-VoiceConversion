from glob import glob
from tqdm import tqdm
import re
import os

import soundfile as sf
import pyworld as pw
import librosa

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import fairseq
from speechbrain.pretrained import EncoderClassifier
from dataset.dataset_info import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from utils.utils import makedirs

""" Speaker Embedder """
import model.speakerEmbedder as spkEmbedder
from model.speakerEmbedder.SpeakerNet import SpeakerNet
from model.textlesslib.textless.data.speech_encoder import SpeechEncoder
from speechbrain.pretrained import EncoderClassifier


import yaml


unit2discreteModel = SpeechEncoder.by_name(
                dense_model_name="hubert-base-ls960",
                quantizer_model_name="kmeans",
                vocab_size=100,
                deduplicate=False,
).to(device)


""" Load speaker embedder """
# before training, we get speaker embedding vector from the pre-trained speaker embedding model
# where the generalized speaker vector outputs. 
# We use the ECAPA-TDNN from HUGGINGFASE.
def get_speaker_embedder():    
    dim_spk = spkEmbedder.kwarg_SPEAKER['nOut']
    
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    classifier.eval()
    print("SpeakerNet is ready.")
    
    return classifier

# def get_speaker_embedder():    
#     dim_spk = spkEmbedder.kwarg_SPEAKER['nOut']
    
#     m_info = torch.load(
#         "./model/speakerEmbedder/baseline_v2_smproto.model", 
#         map_location=device
#     )                                                                   #=== Load pretrained speaker embedder model
    
#     spk_model = SpeakerNet(**spkEmbedder.kwarg_SPEAKER).to(device)      #=== SpeakerNet     
#     spk_model.eval()
    
#     spk_model.load_state_dict(m_info, strict=False)                     #=== load state dict
#     print("SpeakerNet is ready.")
    
#     return spk_model

    
class DataPreprocess():
    def __init__(self, config):        
        ROOT = config["Root_Dataset"]
        self.top_db = config["Preprocess"]["top_db"]
        self.sr = config["Preprocess"]["sr"]

        self.dict_path_dataset = {
            'ESD': ROOT + config['ESD']["Paths"]["path_dataset"],
            'EmovDB': ROOT + config['EmovDB']["Paths"]["path_dataset"],
            'RAVDESS': ROOT + config['RAVDESS']["Paths"]["path_dataset"],
            'SAVEE': ROOT + config['SAVEE']["Paths"]["path_dataset"],
            'JL_Corpus': ROOT + config['JL_Corpus']["Paths"]["path_dataset"],
        }
        
        self.dict_savePath_dataset = {
            'ESD': ROOT + config['ESD']["Paths"]["path_save"],
            'EmovDB': ROOT + config['EmovDB']["Paths"]["path_save"],
            'RAVDESS': ROOT + config['RAVDESS']["Paths"]["path_save"],
            'SAVEE': ROOT + config['SAVEE']["Paths"]["path_save"],
            'JL_Corpus': ROOT + config['JL_Corpus']["Paths"]["path_save"],
        }
        
        self.dict_emotion_lists = {
            'ESD': config['ESD']['list_emotions'],
            'EmovDB': config['EmovDB']['list_emotions'],
            'RAVDESS': config['RAVDESS']['list_emotions'],
            'SAVEE': config['SAVEE']['list_emotions'],
            'JL_Corpus': config['JL_Corpus']['list_emotions'],
        }
        
        assert set(self.dict_path_dataset.keys()) == set(self.dict_savePath_dataset.keys()), 'DataPreprocess'
        self.list_data_available = self.dict_path_dataset.keys()
        
    @property
    def dataName(self):
        return self.__dataName
    
    @dataName.setter
    def dataName(self, dataName):
        if dataName not in self.list_data_available:
            raise ValueError('DataName is not available for preprocessing a dataset.')
        
        self.__dataName = dataName
        self.__path_dataset = self.dict_path_dataset[dataName]
        self.__path_saveData = self.dict_savePath_dataset[dataName]
        
        print("Data Name: {}".format(self.__dataName))
        print("Path of Dataset: {}".format(self.__path_dataset))
        print("Path to Save: {}".format(self.__path_saveData))
        
        
    def preprocess(self, pretrainedSpkModel, speech2unitModel):
        """
        OUTPUT 
        ------
        * trimmed wave      || numpy
            top_db is set by 20.
        
        """
        pth_wave_list = self._gather_all_paths()
        print(pth_wave_list)
        
        pbar = tqdm(pth_wave_list)
        for pth in pbar:
            #=== Get Data
            wav_npy, _ = librosa.load(pth, sr=self.sr)
            
            _, emo_state, file_name = self._get_info_from_pth(pth.replace(self.__path_dataset, ""))
            pbar.set_postfix({'Path' : file_name})
            
            
            #=== trim waveform
            wav_npy = librosa.effects.trim(wav_npy, top_db=self.top_db)[0]
            wav_torch = torch.tensor(wav_npy).unsqueeze(0).float().to(device)
            wav_torch = F.pad(wav_torch, ((400 - 320) // 2, (400 - 320) // 2), "reflect")
                
            #=== get embedding (continuous)
            inputs = {
                    "source": wav_torch,
                    "padding_mask": torch.BoolTensor(wav_torch.shape).fill_(False).to(device),
                    "output_layer": 12,  # layer 12
            }
                
            with torch.no_grad():
                unit_emb = speech2unitModel.extract_features(**inputs)[0]
                unit_emb_npy = unit_emb.numpy(force=True)

            #=== get speaker embedding
            with torch.no_grad():
                spk_emb = pretrainedSpkModel.encode_batch(wav_torch)     # (1, 1, dim_spk = 192)
                spk_emb_npy = spk_emb.squeeze(0).squeeze(0).numpy(force=True)
            
            
            #=== Save trimmed waveform and speaker embedding
            _dir_wav = os.path.join(self.__path_saveData, 'wav')
            _dir_unit = os.path.join(self.__path_saveData, 'unit')
            _dir_spkEmb = os.path.join(self.__path_saveData, 'spkEmb')
            
            if not os.path.exists(_dir_wav):
                makedirs(_dir_wav)          # If there's no dir, then make it!
            if not os.path.exists(_dir_spkEmb):
                makedirs(_dir_spkEmb)       # If there's no dir, then make it!
            if not os.path.exists(_dir_unit):
                makedirs(_dir_unit)         # If there's no dir, then make it!
                
            sf.write(os.path.join(_dir_wav, file_name.replace('.npy', '.wav')), wav_npy, self.sr)
            np.save(os.path.join(_dir_wav, file_name), wav_npy)                 # (1, wav_length)
            np.save(os.path.join(_dir_unit, file_name), unit_emb_npy)           # (1, frame_length, dim)
            np.save(os.path.join(_dir_spkEmb, file_name), spk_emb_npy)          # (1, dim_spk)

            
    def _get_info_from_pth(self, pth):
        if self.__dataName == 'ESD':
            return ESD_str_to_info(pth)
        
        elif self.__dataName == 'EmovDB':
            return EmovDB_str_to_info(pth)
        
        elif self.__dataName == 'RAVDESS':
            return RAVDESS_str_to_info(pth)
        
        elif self.__dataName == 'SAVEE':
            return SAVEE_str_to_info(pth)
        
        elif self.__dataName == 'JL_Corpus':
            return JL_Corpus_str_to_info(pth)

            
    def _gather_all_paths(self):
        """
            Gether all paths of each data as _pth_glob.
        """        
        if self.__dataName == 'ESD':
            path_dataset = self.dict_path_dataset['ESD']
            
            _pth_glob = "{}/**/**/*".format(path_dataset)        #=== ex. ROOT/0018/Surprise/0018_001417.wav'
            pth_wave_list = glob(_pth_glob)
            pth_wave_list = [pth for pth in pth_wave_list if int(re.sub('^0.', "", pth.split("/")[-3])) > 10]
            
        elif self.__dataName == 'EmovDB':
            path_dataset = self.dict_path_dataset['EmovDB']          #==== ex. ROOT/bea/amused_1-15_0010.wav"
            
            _pth_glob = "{}/**/*".format(path_dataset)
            pth_wave_list = glob(_pth_glob)
            pth_wave_list = [pth for pth in pth_wave_list if pth.split("/")[-1].split("_")[0] in self.dict_emotion_lists['EmovDB']]
            
        elif self.__dataName == 'RAVDESS':
            path_dataset = self.dict_path_dataset['RAVDESS']
            
            _pth_glob = "{}/Actor_*/*".format(path_dataset)         #==== ex. ROOT/Actor_21/03-01-02-02-01-01-21.wav"

            pth_wave_list = glob(_pth_glob)
            pth_wave_list = [pth for pth in pth_wave_list if not(pth.split('-')[3] == '01' and pth.split('-')[2] != '01') and pth.split('-')[2] in self.dict_emotion_lists['RAVDESS']]
            
        elif self.__dataName == 'SAVEE':
            path_dataset = self.dict_path_dataset['SAVEE']
            
            _pth_glob = "{}/*".format(path_dataset)                 #=== ex. ROOT/DC_a12.wav
            pth_wave_list = glob(_pth_glob)
            
        elif self.__dataName == 'JL_Corpus':
            path_dataset = self.dict_path_dataset['JL_Corpus']       
            
            _pth_glob = "{}/*.wav".format(path_dataset)             #=== ex. ROOT/female2_angry_5b_2.wav
            pth_wave_list = glob(_pth_glob)
            pth_wave_list = [pth for pth in pth_wave_list if pth.split('/')[-1].split('_')[1] in self.dict_emotion_lists['JL_Corpus']]
        
        return pth_wave_list
        
        


if __name__ == "__main__":
    config = yaml.load(
        open("./config/config_preprocess.yaml", "r"), Loader=yaml.FullLoader
    )

    
    
    contentVec_ckpt_path = config['Model']['Pretrained']['ContentVec']['model_path']
    
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([contentVec_ckpt_path])
    speech2unitModel = models[0]            # Context Vec Model
    speech2unitModel.to(device)
    speech2unitModel.eval()
    
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
    
    preprocessing = DataPreprocess(config)
    
    #=== ESD
    # preprocessing.dataName = 'ESD'
    # preprocessing.preprocess(classifier, speech2unitModel)
    
    #=== EmovDB
    preprocessing.dataName = 'EmovDB'
    preprocessing.preprocess(classifier, speech2unitModel)
    
    # #=== RAVDESS
    # preprocessing.dataName = 'RAVDESS'
    # preprocessing.preprocess(classifier, speech2unitModel)
    
    # #=== SAVEE
    # preprocessing.dataName = 'SAVEE'
    # preprocessing.preprocess(classifier, speech2unitModel)
    
    # #=== JL_Corpus
    # preprocessing.dataName = 'JL_Corpus'
    # preprocessing.preprocess(classifier, speech2unitModel)



