import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import torch.nn.utils.weight_norm as wn

from .layers import *


""" Encoder """

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        """ Parameter """
        mode_spec_input = config['Model']['use_spec_input']
        dim_mel = config['Dataset']['dim_mel']
        
        dim_spec = config['Dataset']['n_fft'] // 2 + 1 if mode_spec_input else dim_mel
        
        n_Block = config['Model']['VAE']['n_EncVCBlock']
        n_emo = config['Model']['n_emo']
        dim_hid = config['Model']['VAE']['dim_encoder_hidden']
        
        dropout = config['Model']['VAE']['dropout_encoder']
        
        
        """ Architecture """
        # 1) Pre-Linear Layer
        # self.prenet = nn.Sequential(
        #     nn.Linear(dim_spec, dim_hid, bias=False),
        #     # nn.GELU(),
        #     # nn.Dropout(dropout),
        #     # nn.Conv1d(dim_hid, dim_hid, bias=False, kernel_size=3),
        #     # nn.GELU(),
        #     # nn.Dropout(dropout)
        # )
        
        in_channels = dim_hid * 8 + dim_spec
        
        self.prenet = nn.Sequential(
            ConvBank(dim_spec, dim_hid), 
            nn.Conv1d(in_channels, dim_hid, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )    
        
        # 2) Encoder-VQ Blocks
        self.Blocks = nn.ModuleList([
            EncVCBlock(config) for _ in range(n_Block)
        ])
        
        # 3) Encoder-VQ Blocks (linear)
        self.linear_blocks = nn.ModuleList([
            nn.Sequential(
                wn(nn.Linear(dim_hid, dim_hid)),
                nn.GELU(),
                wn(nn.Linear(dim_hid, dim_hid)),
                nn.GELU()
            ) for _ in range(6)
        ])
        
        self.unshared_mu = nn.ModuleList()
        self.unshared_logvar = nn.ModuleList()
        for _ in range(n_emo):
            self.unshared_mu += [nn.Linear(dim_hid, dim_hid, bias=False)] 
            self.unshared_logvar += [nn.Linear(dim_hid, dim_hid, bias=False)] 
        
        # self.mu = nn.Linear(dim_hid, dim_hid, bias=False)
        # self.logvar = nn.Linear(dim_hid, dim_hid, bias=False)
        
        
    def forward(self, spec, emo_state):
        """
            #=== INPUT
            * wav: trimmed raw-waveform
        """
        
        #== Pre-Net
        hid = self.prenet(spec).transpose(1, 2)
        
        #== Encoder Blocks      
        for block in self.Blocks:
            hid = block(hid)
            
        #== Then, N-Avg Pooling
        hid = hid.mean(1)
        
        #== Linear
        for layer in self.linear_blocks:
            hid = layer(hid) + hid
            
        # return self.mu(hid), self.logvar(hid)
            
        out_mu = []
        out_logvar = []
        for layer_mu, layer_logvar in zip(self.unshared_mu, self.unshared_logvar):
            out_mu += [layer_mu(hid)]
            out_logvar += [layer_logvar(hid)]
            
        out_mu = torch.stack(out_mu, dim=1)                           # (batch, num_domains, style_dim)
        out_logvar = torch.stack(out_logvar, dim=1)                           # (batch, num_domains, style_dim)
        
        idx = torch.LongTensor(range(emo_state.size(0))).to(emo_state.device)
        
        out_mu = out_mu[idx, emo_state]  # (batch, style_dim)
        out_logvar = out_logvar[idx, emo_state]  # (batch, style_dim)
        
        return out_mu, out_logvar
    
    
    def reparam(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + eps * std




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
        down_scale = 1
    ):
        super().__init__()
        
        """ Parameter """
        self.down_scale = down_scale
        
        """ Architecture """
        self.conv1 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, use_norm=False)
        self.conv2 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, use_norm=False)
        self.conv3 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, use_norm=False)
        self.conv4 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, stride=down_scale, use_norm=False)
        
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


