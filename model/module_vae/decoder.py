import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
from .fftBlock import FFTBlock

from .layers import *


""" Decoder """

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim_enc_hid = config['Model']['VAE']['dim_encoder_hidden']
        dim_hid = config['Model']['VAE']['dim_decoder_hidden']
        
        """ Architecture """
        #=== unit embedding
        n_units = config['Model']['Pretrained']['HuBERT']['vocab_size']
        self.mode_unit_discrete = config['Train']['mode_unit_discrete']
        
        if self.mode_unit_discrete:
            self.unit_embedding = nn.Embedding(config['Model']['Pretrained']['HuBERT']['vocab_size'], dim_hid)
        else:
            self.unit_embedding = nn.Linear(768, dim_hid)
        
        #=== Decoder-VQ Blocks
        n_Block = config['Model']['VAE']['n_DecVCBlock']
        list_upscale = config['Model']['VAE']['list_upscale']
        
        self.Blocks = nn.ModuleList([
            DecVCBlock(config, list_upscale[i]) for i in range(n_Block)
        ])
        
        self.linear_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_enc_hid, dim_enc_hid),
                nn.GELU(),
                nn.Linear(dim_enc_hid, dim_enc_hid),
                nn.GELU()
            ) for _ in range(4)
        ])
        
        #=== last linear
        dim_mel = config['Loader']['dim_mel']
        
        n_head = 2
        self.fft_block_2 = FFTBlock(dim_hid, n_head, dim_hid//2, dim_hid//2, dim_hid*4, [9, 1])
        self.fft_block_3 = FFTBlock(dim_hid, n_head, dim_hid//2, dim_hid//2, dim_hid*4, [9, 1])
        
        self.post_linear_1 = nn.Linear(dim_hid, dim_mel, bias=False)
        self.post_linear_2 = nn.Linear(dim_hid, dim_mel, bias=False)
        self.post_linear_3 = nn.Linear(dim_hid, dim_mel, bias=False)
        
        
    def forward(self, unit, z_style):
        """
        #=== INPUT
        * unit          || torch([B, T])
        * z_style       || torch([B, dim_latent])
        """
        
        hid = self.unit_embedding(unit)      # (B, T, dim_unit) -> (B, T, dim_hid)
        #hid = unit
       
        cond = z_style
        for layer in self.linear_blocks:
            cond = layer(cond) + cond
            
        for layer in self.Blocks:
            hid = layer(hid, cond)
            
        mels = [self.post_linear_1(hid).contiguous().transpose(1, 2)]       # (B, C, T)
        
        hid, _ = self.fft_block_2(hid)
        mels.append(self.post_linear_2(hid).contiguous().transpose(1, 2))   # (B, C, T)
        
        hid, _ = self.fft_block_3(hid)
        mels.append(self.post_linear_3(hid).contiguous().transpose(1, 2))   # (B, C, T)

        return mels  
    


class DecVCBlock(nn.Module):
    def __init__(self, config, up_scale):
        super().__init__()
        
        """ Model Parameter """
        dim_latent = config['Model']['Style_Prior']['CNF']['dim_latent']
        d_hid = config['Model']['VAE']['dim_decoder_hidden']
        
        kernel_size = config['Model']['VAE']['kernel_size_decoder']
        dropout = config['Model']['VAE']['dropout_decoder']
        
        """ Architecture """
        # Convolution Module
        self.conv_module = DecConvModule(d_hid, dim_latent, kernel_size, dropout, up_scale)
        
    def forward(self, x, cond):
        """
        ? INPUT
        :input: tensor, (B, T, C)
        :quant_emb: tensor, (B, T, C_quant)
            This feature is invariant for timbre, and pitch, which is quantized from dictionarys.
        :spk_emb: tensor, (B, C_spk)
        :emo_id: int tensor, (B, C_emb)
        :qz_stats: list, [qz_mean, qz_std]
        """
        hid = x
        
        #== Convolution Module
        hid = self.conv_module(hid, cond)
        
        return hid




class DecConvModule(nn.Module):
    def __init__(
        self,
        d_hid,
        d_style,
        kernel_size,
        dropout,
        up_scale = 2,
    ):
        super().__init__()
        
        """ Parameter """
        self.up_scale = up_scale
        
        """ Architecture """
        if up_scale != 1:
            self.up_scale = up_scale

        self.conv1 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, d_cond=d_style)
        self.conv2 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, d_cond=d_style)
        self.conv3 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, d_cond=d_style)
        self.conv4 = UpConv(d_hid, d_hid, kernel_size, dropout=dropout, d_cond=d_style, up_scale=up_scale)
        
    def forward(self, x, spk_emb):
        """
        ? INPUT
        :input tensor: tensor, (B, T, C)
        :spk_emb: tensor, (B, C_style)
        """
        
        #== residual & 2 conv
        hid = self.conv2(self.conv1(x, spk_emb), spk_emb)
        hid = x + hid
        
        #== residual (upsample) & 2 conv (with PixelShuffle)
        out = self.conv4(self.conv3(hid, spk_emb), spk_emb)
        if self.up_scale != 1:
            out = out + self._upsample(hid, self.up_scale) 
        else:
            out = out + hid

        return out
    
    def _upsample(self, emb, up_scale):
        emb = emb.transpose(1, 2)
        return F.interpolate(emb, scale_factor=up_scale, mode='nearest').contiguous().transpose(1, 2)



if __name__ == "__main__":
    import yaml
    config = yaml.load(
        open("../config/config.yaml", "r"), Loader=yaml.FullLoader
    )

    test_mel = torch.zeros(8, 128, 80)
    test_spk_id = torch.randint(0, 10, (8,))
    test_emo_id = torch.randint(0, 5, (8,))

    model = VoiceConversionModel(config)

    from torchsummaryX import summary
    summary(model, test_mel, spk_id=test_spk_id, emo_id=test_emo_id)


