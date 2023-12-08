import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

from .layers import *


""" Encoder """

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        """ Parameter """
        dim_spec = config['Loader']['n_fft'] // 2 + 1
        
        n_Block = config['Model']['VAE']['n_EncVCBlock']
        dim_hid = config['Model']['VAE']['dim_encoder_hidden']
        
        dropout = config['Model']['VAE']['dropout_encoder']
        
        
        """ Architecture """
        # 1) Pre-Linear Layer
        self.prenet = nn.Sequential(
            nn.Linear(dim_spec, dim_hid, bias=False),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Conv1d(dim_hid, dim_hid, bias=False, kernel_size=3),
            # nn.GELU(),
            # nn.Dropout(dropout)
        )
        
        # 2) Encoder-VQ Blocks
        self.Blocks = nn.ModuleList([
            EncVCBlock(config) for _ in range(n_Block)
        ])
        
        # 3) Encoder-VQ Blocks (linear)
        self.linear_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hid, dim_hid),
                nn.GELU(),
                nn.Linear(dim_hid, dim_hid),
                nn.GELU()
            ) for _ in range(2)
        ])
        
        
    def forward(self, spec):
        """
            #=== INPUT
            * wav: trimmed raw-waveform
        """
        
        #== Pre-Net
        hid = self.prenet(spec.transpose(1, 2))
        
        #== Encoder Blocks      
        for block in self.Blocks:
            hid = block(hid)
            
        #== Then, N-Avg Pooling
        hid = hid.mean(1)
        
        for layer in self.linear_blocks:
            hid = layer(hid) + hid
        
        #== Speaker Module
        return hid




""" Voice Conversion Blocks """

class EncVCBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        """ Parameter """
        d_hid = config['Model']['VAE']['dim_encoder_hidden']
        
        kernel_size = config['Model']['VAE']['kernel_size_encoder']
        dropout = config['Model']['VAE']['dropout_encoder']
    
        """ Architecture """
        # Convolution Module
        self.conv_module = EncConvModule(d_hid, kernel_size, dropout)
        
        
    def forward(self, x):
        """ (B, T, C) -> (B, T, C) """
        
        out = self.conv_module(x)
        
        return out
        
        
 

""" Convolution Module """

class EncConvModule(nn.Module):
    def __init__(
        self,
        d_hid,
        kernel_size,
        dropout,
        down_scale = 2
    ):
        super().__init__()
        
        """ Parameter """
        self.down_scale = down_scale
        
        """ Architecture """
        self.conv1 = Conv(d_hid, d_hid, kernel_size, dropout=dropout)
        self.conv2 = Conv(d_hid, d_hid, kernel_size, dropout=dropout)
        self.conv3 = Conv(d_hid, d_hid, kernel_size, dropout=dropout)
        self.conv4 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, stride=down_scale)
        
    def forward(self, x):
        #== Residual & 2 conv
        hid = self.conv2(self.conv1(x))
        hid = hid + x
        
        #== Residual + 2 conv
        out = self.conv4(self.conv3(hid))
        out = self._downsample(hid, self.down_scale) + out
        
        return out
    
    def _downsample(self, x, down_scale):
        return F.avg_pool1d(x.contiguous().transpose(1, 2), kernel_size=down_scale).contiguous().transpose(1, 2)
        






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


