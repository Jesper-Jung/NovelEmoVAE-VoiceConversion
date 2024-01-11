import sys
sys.path.append("..")

import os
from utils.dataset_utils import get_mel_from_audio
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.distributions import Normal

torch.manual_seed(44)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np

import argparse
import time
import fairseq

from dataset import Dataset_ESD
from torch.utils.data import DataLoader

from model import EmotionStyleGenerationFlowVAE as ModelStructure

from utils import check_recon_mel, makedirs
import soundfile as sf
import yaml

contentVec_ckpt_path = "./checkpoint_best_legacy_500.pt"


""" Configuration """

def _argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--best_path', type=str,
        default="./assets/220320/1702915131.165659"        
    )
    parser.add_argument(
        '--saved_model', type=str,
        default="checkpoint_120000.pth.tar"      # type example: ""
    )
    
    args = parser.parse_args()
    return args

def get_wave_from_mel(mel, pth_saving, mode="recon", sr=16000):
    vocoder = torch.hub.load("bshall/hifigan:main", "hifigan", trust_repo=True).to(device)
    # Load Model

    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)
        
    wav = vocoder(mel)

    sf.write(pth_saving + f"/{mode}.wav", np.ravel(wav[0].numpy(force=True)), sr)




class Synthesizer():
    def __init__(self, args):
        """ Model & Checkpoint """
        # Checkpoint
        ckpt_pth = os.path.join(args.best_path, args.saved_model)
        ckpt = torch.load(ckpt_pth, map_location=device)

        # Yaml Config
        self.config = yaml.load(open(os.path.join(args.best_path, "config.yaml")), Loader=yaml.FullLoader)
        self.mode_unit_discrete = self.config["Train"]["mode_unit_discrete"]
        self.dim_latent = self.config["Model"]["Style_Prior"]["CNF"]["dim_latent"]
        
        # Model
        self.model = ModelStructure(self.config).to(device)
        self.model.eval()

        self.model.load_state_dict(ckpt['model'])

        # Test Dataset
        config_ESD = yaml.load(
            open("./config/ESD_dataset.yaml", "r"), Loader=yaml.FullLoader
        )

        self.dataset = Dataset_ESD(config_ESD, 'test')



    def test_step(self, src_wav, tar_wav):
        from model.speakerEmbedder.SpeakerNet import SpeakerNet
        import model.speakerEmbedder as spkEmbedder
        dim_spk = spkEmbedder.kwarg_SPEAKER['nOut']
        
        with torch.no_grad():
            ###==== 1) Inference speaker embedding from target wav
            print("Inference speaker embedding")
            spk_model = SpeakerNet(**spkEmbedder.kwarg_SPEAKER).to(device)      #=== SpeakerNet     
            
            # Load pre-trained model
            m_info = torch.load("./model/speakerEmbedder/baseline_v2_smproto.model", map_location=device)
            
            # State dict
            spk_model.load_state_dict(m_info, strict=False)
            spk_model.eval()
            print(tar_wav.shape)
            tar_spk_emb = spk_model(tar_wav)
            
            
            ###==== 2) Inference unit embedding from target wav
            from model.textlesslib.textless.data.speech_encoder import SpeechEncoder
            
            print("Inference unit embedding")
            unit2discreteModel = SpeechEncoder.by_name(
                dense_model_name="hubert-base-ls960",
                quantizer_model_name="kmeans",
                vocab_size=100,
                deduplicate=False,
            ).to(device)
            
            if self.mode_unit_discrete:
                src_wav = F.pad(src_wav, ((400 - 320) // 2, (400 - 320) // 2), "reflect")
                src_unit = unit2discreteModel(src_wav)["units"].unsqueeze(0)
                src_unit = src_unit.repeat(7, 1)
                #src_unit = src_unit[:, 0:src_unit.shape[-1]//2].repeat_interleave(2, dim=-1)
            else:
                inputs = {
                    "source": src_wav.to(device),
                    "padding_mask": torch.BoolTensor(src_wav.shape).fill_(False).to(device),
                    "output_layer": 12,  # layer 12
                }
                
                models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([contentVec_ckpt_path])
                speech2unitModel = models[0]            # Context Vec Model
                speech2unitModel.to(device)
                speech2unitModel.eval()
                
                src_unit = speech2unitModel.extract_features(**inputs)[0]
                src_unit = src_unit.repeat(7, 1, 1)
                
                
                
            
            ###==== 3) get style vector from prior network
            noise = Normal(0, 1).sample((7, self.dim_latent)).to(device)
            tar_spk_emb = tar_spk_emb.repeat(7, 1)
            tar_emoID = torch.arange(7).to(torch.int64).to(tar_spk_emb)
            print(noise.shape, tar_spk_emb.shape, tar_emoID.shape)
            
            z_style = self.model.style_prior(noise, tar_spk_emb, tar_emoID, reverse=True)   
            
            
            ###==== 4) synthesize with the unit embedding and the style vector through the decoder.
            
            _, _, converse_mel = self.model.adain_decoder(src_unit, z_style)       # (B, C, T)
            
            
            #== wav to melspectrogram
            from model.module_vae.layers import LogMelSpectrogram, Spectrogram
            
            self.log_mel_transform = LogMelSpectrogram(self.config).to(device)
            self.log_mel_transform.eval()
            src_mel = self.log_mel_transform(src_wav).detach()
            tar_mel = self.log_mel_transform(tar_wav).detach()
            # print(torch.nn.CosineSimilarity()(src_mean, tar_mean))


        torch_list = [*converse_mel, src_mel, tar_mel]
        npy_list = [
            converse_mel[0].squeeze(0).numpy(force=True), 
            converse_mel[1].squeeze(0).numpy(force=True), 
            converse_mel[2].squeeze(0).numpy(force=True), 
            converse_mel[3].squeeze(0).numpy(force=True), 
            converse_mel[4].squeeze(0).numpy(force=True), 
            converse_mel[5].squeeze(0).numpy(force=True), 
            converse_mel[6].squeeze(0).numpy(force=True), 
            src_mel.squeeze(0).numpy(force=True), 
            tar_mel.squeeze(0).numpy(force=True)
        ]
        
        print("Forward End!")

        return torch_list, npy_list

            # ### Convert
            # converse_mel, _ = self.model.decoder(src_VQlist, ran_emb, src_emoID)
            # converse_mel = self._mel_denormalize(converse_mel[0])
            # converse_mel_npy = converse_mel.to('cpu').detach().numpy().T

            # recon_mel, _ = self.model.decoder(src_VQlist, src_spk_emb, src_emoID)
            # recon_mel = self._mel_denormalize(recon_mel[0])
            # recon_mel_npy = recon_mel.to('cpu').detach().numpy().T

            # src_mel = self._mel_denormalize(src_mel[0])
            # src_mel_npy = src_mel.to('cpu').detach().numpy().T

            # tar_mel = self._mel_denormalize(tar_mel[0])
            # tar_mel_npy = tar_mel.to('cpu').detach().numpy().T
            # ### ==================================================================

        


    def conversion_seen_to_seen(self, src_ind, tar_ind):
        # Make directory to save
        self._makedir()

        """ Data Preparing """
        src_wav, tar_wav = self.prepare_from_ESD(src_ind, tar_ind)        

        """ Forward """
        torch_list, npy_list = self.test_step(src_wav, tar_wav)
        
        *converse_mel, src_mel, tar_mel = torch_list
        *converse_mel_npy, src_mel_npy, tar_mel_npy = npy_list

        # Check Mel
        check_recon_mel(converse_mel_npy[0], self.test_path, 0, mode='converse0')
        check_recon_mel(converse_mel_npy[1], self.test_path, 0, mode='converse1')
        check_recon_mel(converse_mel_npy[2], self.test_path, 0, mode='converse2')
        check_recon_mel(converse_mel_npy[3], self.test_path, 0, mode='converse3')
        check_recon_mel(converse_mel_npy[4], self.test_path, 0, mode='converse4')
        check_recon_mel(converse_mel_npy[5], self.test_path, 0, mode='converse5')
        check_recon_mel(converse_mel_npy[6], self.test_path, 0, mode='converse6')
        check_recon_mel(src_mel_npy, self.test_path, 0, mode='GT')
        check_recon_mel(tar_mel_npy, self.test_path, 0, mode='target')

        # Get wav
        get_wave_from_mel(converse_mel[0], self.test_path, mode='converse0', sr=16000)
        get_wave_from_mel(converse_mel[1], self.test_path, mode='converse1', sr=16000)
        get_wave_from_mel(converse_mel[2], self.test_path, mode='converse2', sr=16000)
        get_wave_from_mel(converse_mel[3], self.test_path, mode='converse3', sr=16000)
        get_wave_from_mel(converse_mel[4], self.test_path, mode='converse4', sr=16000)
        get_wave_from_mel(converse_mel[5], self.test_path, mode='converse5', sr=16000)
        get_wave_from_mel(converse_mel[6], self.test_path, mode='converse6', sr=16000)
        get_wave_from_mel(src_mel, self.test_path, mode='GT', sr=16000)
        get_wave_from_mel(tar_mel, self.test_path, mode='target', sr=16000)



    def Converse_custom_to_custom(self, src_path, tar_path):
        ### read audio
        import torchaudio.functional as F
        
        src_audio, src_fs = torchaudio.load(src_path)
        tar_audio, tar_fs = torchaudio.load(tar_path)
        
        print(src_audio.shape, tar_audio.shape)
        
        ###==== stereo to mono
        if src_audio.shape[0] == 2:
            src_audio = src_audio.mean(0, keepdim=True)

        if tar_audio.shape[0] == 2:
            tar_audio = tar_audio.mean(0, keepdim=True)

        # assert src_fs == tar_fs == 44100, "[Synthesizer.py] You need to prepare an audio of rate 16kHz"

        # Downsample
        src_audio = F.resample(src_audio, src_fs, 16000).to(device)
        tar_audio = F.resample(tar_audio, tar_fs, 16000).to(device)
        print(src_audio.shape, tar_audio.shape)
        # src_audio, tar_audio = src_audio[::2], tar_audio[::2]

        ### convert to mel
        src_len = (src_audio.shape[-1] // 320) * 320
        tar_len = (tar_audio.shape[-1] // 320) * 320
        
        src_wav = src_audio[:, :src_len]
        tar_wav = tar_audio[:, :tar_len]

        #print(src_len, tar_len)


        """ Step """

        # Build directory to save
        self._makedir()

        torch_list, npy_list = self.test_step(src_wav, tar_wav)
        
        *converse_mel, src_mel, tar_mel = torch_list
        *converse_mel_npy, src_mel_npy, tar_mel_npy = npy_list

        # Check Mel
        check_recon_mel(converse_mel_npy[0], self.test_path, 0, mode='converse0')
        check_recon_mel(converse_mel_npy[1], self.test_path, 0, mode='converse1')
        check_recon_mel(converse_mel_npy[2], self.test_path, 0, mode='converse2')
        check_recon_mel(converse_mel_npy[3], self.test_path, 0, mode='converse3')
        check_recon_mel(converse_mel_npy[4], self.test_path, 0, mode='converse4')
        check_recon_mel(converse_mel_npy[5], self.test_path, 0, mode='converse5')
        check_recon_mel(converse_mel_npy[6], self.test_path, 0, mode='converse6')
        check_recon_mel(src_mel_npy, self.test_path, 0, mode='GT')
        check_recon_mel(tar_mel_npy, self.test_path, 0, mode='target')

        # Get wav
        get_wave_from_mel(converse_mel[0], self.test_path, mode='converse0', sr=16000)
        get_wave_from_mel(converse_mel[1], self.test_path, mode='converse1', sr=16000)
        get_wave_from_mel(converse_mel[2], self.test_path, mode='converse2', sr=16000)
        get_wave_from_mel(converse_mel[3], self.test_path, mode='converse3', sr=16000)
        get_wave_from_mel(converse_mel[4], self.test_path, mode='converse4', sr=16000)
        get_wave_from_mel(converse_mel[5], self.test_path, mode='converse5', sr=16000)
        get_wave_from_mel(converse_mel[6], self.test_path, mode='converse6', sr=16000)
        get_wave_from_mel(src_mel, self.test_path, mode='GT', sr=16000)
        get_wave_from_mel(tar_mel, self.test_path, mode='target', sr=16000)




    def prepare_from_ESD(self, src_ind, tar_ind):
        src_wav = torch.tensor(self.dataset[src_ind]['wav']).float().to(device)
        tar_wav = torch.tensor(self.dataset[tar_ind]['wav']).float().to(device)

        src_len = (src_wav.shape[0] // 160) * 160
        tar_len = (tar_wav.shape[0] // 160) * 160
        
        src_wav = src_wav[:src_len].unsqueeze(0)
        tar_wav = tar_wav[:tar_len].unsqueeze(0)


        print(src_len, tar_len)
        print(self.dataset[src_ind]['wav'])

        return src_wav, tar_wav


    def _makedir(self):
        ts = time.time()
        self.test_path = os.path.join("./assets_test", str(ts))
        makedirs(self.test_path)

    def _mel_denormalize(self, mel):
        if isinstance(mel, torch.Tensor):
            _mean, _std = torch.tensor(self.mel_mean).float().to(device), torch.tensor(self.mel_std).float().to(device)
            return mel * _std + _mean
        else:
            return mel * self.mel_std + self.mel_mean

    def _mel_normalize(self, mel):
        if isinstance(mel, torch.Tensor):
            _mean, _std = torch.tensor(self.mel_mean).float().to(device), torch.tensor(self.mel_std).float().to(device)
            return (mel - _mean) / _std
        else:
            return (mel - self.mel_mean) / self.mel_std

   # def _M4a2Wav(self, m4a_path):



    
args = _argparse()
synthesizer = Synthesizer(args)

dir_custom = "./data/Dataset_Custom/Custom"

synthesizer.conversion_seen_to_seen(5, 40)
# synthesizer.Converse_custom_to_custom(
#     os.path.join(dir_custom, "hamzi_001_eng.wav"),
#     os.path.join(dir_custom, "male2_neutral_5b_2.wav"),
# )


