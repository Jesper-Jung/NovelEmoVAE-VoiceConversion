import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.distributions import Normal
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import math
import numpy as np

from .module_vae.layers import LogMelSpectrogram, Spectrogram
#import speakerEmbedder as spkEmbedder

""" Unit Encoder """
#from textless.data.speech_encoder import SpeechEncoder

""" Continuous Flow """
from module_vae import *
from module_cnf.flow import cnf



class EmotionStyleGenerationFlowVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        """ Architecture """
        #=== 1) wav to melspectrogram
        self.log_mel_transform = LogMelSpectrogram(config)
        self.log_mel_transform.eval()
        
        self.spec_transform = Spectrogram(config)
        self.spec_transform.eval()
        
        
        #=== 2) [Pre-trained] Unit Encoder (Hubert)
        hubert_model_name = config['Model']['Pretrained']['HuBERT']['dense_model_name']
        hubert_quantizer_name = config['Model']['Pretrained']['HuBERT']['quantizer_name']
        hubert_vocab_size = config['Model']['Pretrained']['HuBERT']['vocab_size']
        
        # self.hubert = SpeechEncoder.by_name(
        #     dense_model_name=hubert_model_name,
        #     quantizer_model_name=hubert_quantizer_name,
        #     vocab_size=hubert_vocab_size,
        #     deduplicate=False,
        # ).to(device)
        # self.hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).to('cuda:2')
        # self.hubert.eval()
        
        #=== 3) Speaking Style Predictor (Prior Net, Conditional Continuous Flow)
        dim_latent = config['Model']['Style_Prior']['CNF']['dim_latent']
        dims_cnf = config['Model']['Style_Prior']['CNF']['dims_cnf']
        dim_spk = config['Model']['Style_Prior']['dim_spk']
        dim_emo = config['Model']['Style_Prior']['dim_emo']
        
        self.style_prior = cnf(dim_latent, dims_cnf, dim_spk, dim_emo, 1, config=config)
        
        dim_enc_hid = config['Model']['VAE']['dim_encoder_hidden']
        self.emotion_classifier = nn.Sequential(
            nn.Linear(dim_enc_hid, dim_enc_hid//2),
            nn.ReLU(),
            nn.Linear(dim_enc_hid//2, dim_enc_hid//4),
            nn.ReLU(),
            nn.Linear(dim_enc_hid//4, 7)
        )
        
        
        #=== 4) Posterior Style Encoder (AdaIN-like)
        self.C_regularization = config['Model']['Posterior']['variance_regularization']
        self.flow_detach = config['Train']['mode_flow_detach']
        
        self.mode_spec = config['Model']['use_spec_input']
        self.adain_encoder = encoder.Encoder(config)
        if not self.flow_detach:
            self.mu_linear = nn.Linear(dim_enc_hid, dim_enc_hid, bias=False)
            self.logvar_linear = nn.Linear(dim_enc_hid, dim_enc_hid, bias=False)
        
        #=== 5) Decoder (AdaIN-like)
        self.adain_decoder = decoder.Decoder(config)
        
        #=== 6) Post-Net
        #self.postnet = PostNet(config)
        
        
    def forward(self, wav, unit, spk_emb, emo_id):
        """
        #=== INPUT
        
        * wav: trimmed,         (batch, trimmed_length)
        * wav_full: non-trimmed, input to spk_embed
        """
        
        batch_size = wav.shape[0]
        
        # 1) Transformation into mel spectrogram
        with torch.no_grad():
            mel_true = self.log_mel_transform(wav).detach()           # (B, C, T)
            spec_true = self.spec_transform(wav).detach()
        
        
        # 2) Output the style embedding from the posterior
        # Here, the variance is regularized at C, since log-Jacovian favours the contraction of the base density,
        # in which the data log-likelihood is fully or partially ignored.
        # See the paper, Conditional Flow Variational Autoencoders for Structured Sequence Prediction
        
        enc_hid = self.adain_encoder(spec_true if self.mode_spec else mel_true)     

        if not self.flow_detach:  
            mu, logvar = self.mu_linear(enc_hid), self.logvar_linear(enc_hid)
            z_style = self.reparam(mu, logvar)
            
            loss_H_post = -0.5 * math.log(2 * math.pi) - 0.5 * logvar - 0.5
            loss_H_post = loss_H_post.sum(-1).mean() / logvar.shape[1]
            
        else:
            z_style = enc_hid
            loss_H_post = torch.tensor([0]).to(wav.device)
            
        #var = torch.zeros_like(mu).fill_(self.C_regularization)
        
        # 3) Forward on continuous flow.
        z_t, delta_log = self.style_prior(
            z_style.detach() if self.flow_detach else z_style, 
            spk_emb, 
            emo_id, 
            torch.zeros(batch_size, 1).to(wav.device)
        )      # (batch_size, dim_noise), (batch_size, 1)
        
        logpz = Normal(0, 1).log_prob(z_t).sum(-1)
        
        delta_log = delta_log.view(batch_size, -1).sum(-1, keepdim=True)
        logpx = logpz - delta_log
        
        loss_flowLL = -logpx.mean()
        loss_flowBPD = loss_flowLL / z_t.shape[1]
        
        emo_pred = self.emotion_classifier(z_style)
        loss_emo_pred = F.cross_entropy(emo_pred, emo_id)
        #loss_emo_pred = torch.tensor([0]).to(device)
        
        
        # 4) reconstruction!
        mels = self.adain_decoder(unit, z_style)       # (B, C, T)
        #mel_post = self.postnet(mel_recon)
        
        loss_recon_1 = self.spec_loss(mel_true, mels[0])
        loss_recon_2 = self.spec_loss(mel_true, mels[1])
        loss_recon_3 = self.spec_loss(mel_true, mels[2])
        
        
        # Return
        loss = [loss_recon_1, loss_recon_2, loss_recon_3, loss_flowBPD, loss_H_post, loss_emo_pred]
        return *mels, mel_true, loss
    
    
    
    # def generation(self, wav, spk_emb, emo_id):
    #     # 1) sample z-vector conditioned on spk_emb and emo_id.
    #     # Dimension of the z-vector is (batch, dim_spk)
        
        
    #     # 2) Get style vector from the cnf.
        
    #     # 3) Get unit represetation from the hubert.
        
    #     # 4) Decode.
        
        

    def set_spkEmbedder(self):
        # Load pre-trained model
        m_info = torch.load("./model/pretrained_model/baseline_v2_smproto.model", map_location=device)
        
        # State dict
        self.spk_embed.load_state_dict(m_info, strict=False)
        self.spk_embed.eval()
        
    def spec_loss(self, mel, pred_mel):
        return F.l1_loss(mel, pred_mel)
    
    def standard_normal_logprob(self, z):
        dim_z = z.size(-1)
        log_z = -0.5 * dim_z * np.log(2 * np.pi)
        return log_z - z.pow(2) / 2
        
    def reparam(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    

""" PostNet """

class PostNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_mel = config["Loader"]["dim_mel"]

        self.conv = nn.ModuleList()

        self.conv.append(
            nn.Sequential(
                nn.Conv1d(n_mel, 512, kernel_size=5, stride=1, padding=2, dilation=1),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.conv.append(
                nn.Sequential(
                    nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
                    nn.BatchNorm1d(512))
            )

        self.conv.append(
            nn.Sequential(
                nn.Conv1d(512, n_mel, kernel_size=5, stride=1, padding=2, dilation=1),
                nn.BatchNorm1d(n_mel))
        )

    def forward(self, x):
        out = x
        for i in range(len(self.conv) - 1):
            out = torch.tanh(self.conv[i](out))

        out = self.conv[-1](out)

        # Residual Connection
        # ! Comment: 의외라 깜짝 놀랐는데 이 한 줄이 loss 입장에서 상당히 중요함.
        out = out + x

        return out
    
    
    