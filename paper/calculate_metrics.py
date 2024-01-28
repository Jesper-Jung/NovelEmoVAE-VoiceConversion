import sys
sys.path.append("..")

from jiwer import wer, cer
from speechbrain.pretrained import SpeakerRecognition as SR
from vocoder_eva.eval import eval_rmse_f0
from dataset import Codebook_EmoState
from transformers import AutoConfig, Wav2Vec2FeatureExtractor



import speech_recognition as sr

import os
from utils.dataset_utils import get_mel_from_audio
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.distributions import Normal
import librosa

torch.manual_seed(44)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np

import argparse
import time
import fairseq

from dataset import EmotionSpeechDataset
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
        default="./assets/220320/our_best"        
    )
    parser.add_argument(
        '--saved_model', type=str,
        default="checkpoint_170000.pth.tar"      # type example: ""
    )
    
    args = parser.parse_args()
    return args

def get_wave_from_mel(mel, pth_saving, mode="recon", sr=16000, return_wav=False):
    vocoder = torch.hub.load("bshall/hifigan:main", "hifigan", trust_repo=True).to(device)
    # Load Model

    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)
        
    wav = vocoder(mel)

    if not return_wav:
        sf.write(pth_saving + f"/{mode}.wav", np.ravel(wav[0].numpy(force=True)), sr)
    else:
        return np.ravel(wav[0].numpy(force=True))
    


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
        config = yaml.load(
            open("./config/config.yaml", "r"), Loader=yaml.FullLoader
        )
        
        config_preprocess = yaml.load(
            open("./config/config_preprocess.yaml", "r"), Loader=yaml.FullLoader
        )

        self.dataset = EmotionSpeechDataset(config, config_preprocess, 'test', synthesizer=True)
        
        
        # tools
        self.r = sr.Recognizer()
        self.verification = SR.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        #https://huggingface.co/harshit345/xlsr-wav2vec-speech-emotion-recognition
        self.vocoder = torch.hub.load("bshall/hifigan:main", "hifigan", trust_repo=True).to(device)
        
        self.dict_results = dict()
    

    def calculation_wer_cer(self, src_wav, tar_wav):
        src_audio = self.numpy_array_to_audio_data(src_wav)
        tar_audio = self.numpy_array_to_audio_data(tar_wav)
        
        with src_audio as source:
            audio = self.r.record(source)
            src_text = self.r.recognize_google(audio)
            print(src_text)
        
        with tar_audio as source:
            audio = self.r.record(source)
            tar_text = self.r.recognize_google(audio)
            print(tar_text)
            
        return self.calculate_wer_cer(src_text, tar_text)
            
        

    def numpy_array_to_audio_data(self, np_array, sample_rate=16000):
        # Write the NumPy array to a buffer as a WAV file
        buffer = io.BytesIO()
        write(buffer, sample_rate, np_array.astype(np.int16))
        buffer.seek(0)

        # Use the buffer as an audio source
        audio_data = sr.AudioFile(buffer)
        return audio_data

    def test_step(self, src_wav, tar_wav):
        from speechbrain.pretrained import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
        
        with torch.no_grad():
            ###==== 1) Inference speaker embedding from target wav
            print("Inference speaker embedding")
            tar_spk_emb = classifier.encode_batch(tar_wav)          # (1, 1, dim_spk = 192)
            tar_spk_emb = tar_spk_emb.squeeze(0)
            
            
            ###==== 2) Inference unit embedding from target wav
            from model.textlesslib.textless.data.speech_encoder import SpeechEncoder
            
            print("Inference unit embedding")

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
            src_unit = src_unit.repeat(5, 1, 1)
                
                
                
            
            ###==== 3) get style vector from prior network
            self.model.eval()
            
            noise = Normal(0, 1).sample((5, self.dim_latent)).to(device)
            tar_spk_emb = tar_spk_emb.repeat(5, 1)
            tar_emoID = torch.arange(5).to(torch.int64).to(tar_spk_emb)
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


    def _calculate_unseen_to_unseen(self, src_ind, tar_ind):
        """ Data Preparing """
        src_pth = self.dataset[src_ind]['pth']
        tar_pth = self.dataset[tar_ind]['pth']
        print(src_pth, tar_pth)
        
        src_emo = self.dataset[src_ind]['emo_state']
        tar_emo = self.dataset[tar_ind]['emo_state']
        print(src_emo, tar_emo)
        
        src_wav, tar_wav = self.prepare_from_ESD(src_ind, tar_ind)        
        
        """ Forward """
        torch_list, npy_list = self.test_step(src_wav, tar_wav)
        
        *converse_mel, src_mel, tar_mel = torch_list
        *converse_mel_npy, src_mel_npy, tar_mel_npy = npy_list
        
        convert_wav = get_wave_from_mel(converse_mel[Codebook_EmoState[tar_emo]], self.test_path, mode='converse0', sr=16000)
        sf.write('./target.wav', tar_wav, fs)
        sf.write('./convert.wav', convert_wav, fs)
        
        
        wer, cer = self.calculation_wer_cer(src_wav, convert_wav)
        _, spk_prediction = verification.verify_files("./target.wav", "./convert.wav")
        spk_prediction = float(spk_prediction)
        eval_rmse_f0()
        
        # self.compute_mcd(tar_wav, convert_wav)
        # self.compute_mcd(_wav, src_wav)
    

    def conversion_unseen_to_unseen(self, src_ind, tar_ind):
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
        check_recon_mel(src_mel_npy, self.test_path, 0, mode='GT')
        check_recon_mel(tar_mel_npy, self.test_path, 0, mode='target')

        # Get wav
        get_wave_from_mel(converse_mel[0], self.test_path, mode='converse0', sr=16000)
        get_wave_from_mel(converse_mel[1], self.test_path, mode='converse1', sr=16000)
        get_wave_from_mel(converse_mel[2], self.test_path, mode='converse2', sr=16000)
        get_wave_from_mel(converse_mel[3], self.test_path, mode='converse3', sr=16000)
        get_wave_from_mel(converse_mel[4], self.test_path, mode='converse4', sr=16000)
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
        check_recon_mel(src_mel_npy, self.test_path, 0, mode='GT')
        check_recon_mel(tar_mel_npy, self.test_path, 0, mode='target')

        # Get wav
        get_wave_from_mel(converse_mel[0], self.test_path, mode='converse0', sr=16000)
        get_wave_from_mel(converse_mel[1], self.test_path, mode='converse1', sr=16000)
        get_wave_from_mel(converse_mel[2], self.test_path, mode='converse2', sr=16000)
        get_wave_from_mel(converse_mel[3], self.test_path, mode='converse3', sr=16000)
        get_wave_from_mel(converse_mel[4], self.test_path, mode='converse4', sr=16000)
        get_wave_from_mel(src_mel, self.test_path, mode='GT', sr=16000)
        get_wave_from_mel(tar_mel, self.test_path, mode='target', sr=16000)

    def prepare_from_ESD(self, src_ind, tar_ind):
        src_wav = torch.tensor(self.dataset[src_ind]['wav']).float().to(device)
        tar_wav = torch.tensor(self.dataset[tar_ind]['wav']).float().to(device)

        src_len = (src_wav.shape[0] // 80) * 80
        tar_len = (tar_wav.shape[0] // 80) * 80
        
        src_wav = src_wav[:src_len].unsqueeze(0)
        tar_wav = tar_wav[:tar_len].unsqueeze(0)


        print(src_len, tar_len)
        print(self.dataset[src_ind]['wav'])
        print("Source: {}".format(self.dataset[src_ind]['pth']))
        print("Target: {}".format(self.dataset[tar_ind]['pth']))

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
        

    def calculate_wer_cer(self, original_transcription, converted_transcription):
        """
        Calculate Word Error Rate (WER) and Character Error Rate (CER).
        
        :param original_transcription: The original transcription as a string.
        :param converted_transcription: The converted transcription as a string.
        :return: A tuple containing the WER and CER.
        """
        # Calculate WER
        word_error_rate = wer(original_transcription, converted_transcription)

        # Calculate CER
        character_error_rate = cer(original_transcription, converted_transcription)

        return word_error_rate, character_error_rate


    def compute_mcd(self, original_audio, converted_audio, sr=16000, n_mfcc=24):
        
        # Compute MFCCs
        mfcc_original = librosa.feature.mfcc(y=original_audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_converted = librosa.feature.mfcc(y=converted_audio, sr=sr, n_mfcc=n_mfcc)

        # Calculate the Euclidean distance and average it
        distances = np.linalg.norm(mfcc_original - mfcc_converted, axis=1)
        mcd = np.mean(distances)

        return mcd


   # def _M4a2Wav(self, m4a_path):



    
args = _argparse()
synthesizer = Synthesizer(args)
print(len(synthesizer.dataset))

dir_custom = "./data/Dataset_Custom/Custom"

synthesizer._calculate_unseen_to_unseen(58, 232)





# synthesizer.Converse_custom_to_custom(
#     os.path.join(dir_custom, "hamzi_001_eng.wav"),
#     os.path.join(dir_custom, "male2_neutral_5b_2.wav"),
# )


