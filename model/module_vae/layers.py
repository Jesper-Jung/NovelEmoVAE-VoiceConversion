import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.utils.spectral_norm as sn
import torch.nn.utils.weight_norm as wn

import torchaudio.transforms as transforms


""" Voice Conversion Layers """


class Conv(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        kernel_size,
        dropout=0.,
        stride=1,
        bias=True,
        d_cond=None,
        use_norm=True
    ):
        super().__init__()
        
        """ Parameter """
        padding = (kernel_size - 1) // 2
        
        """ Architecture """
        self.conv = nn.Conv1d(
            d_in, d_out, kernel_size, 
            padding=padding, padding_mode='replicate', stride=stride, bias=bias
        )
        if not use_norm:
            self.conv = wn(self.conv)
        
        self.dropout = nn.Dropout(dropout)
        
        self.norm = lambda x: x
        if use_norm:
            self.norm = AdaIN(d_out, d_cond) if d_cond is not None else nn.InstanceNorm1d(d_out, affine=False)
        
    def forward(self, x, cond=None):
        out = self.conv(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2)
        out = self.dropout(out)
        out = F.gelu(out)
        
        if isinstance(self.norm, AdaIN):
            out = self.norm(out, cond)
        else:
            out = self.norm(out)

        return self.dropout(out)
    
    
    
class UpConv(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        kernel_size,
        dropout=0.,
        up_scale=2,
        d_cond=None,
    ):
        super().__init__()
        
        """ Parameter """
        padding = (kernel_size - 1) // 2
        
        """ Architecture """
        self.conv = nn.Conv1d(
            d_in, up_scale * d_out, kernel_size, 
            padding=padding, padding_mode='replicate', stride=1
        )
        self.dropout = nn.Dropout(dropout)
        
        self.up_scale = up_scale
        if up_scale != 1:
            self.shuffle = PixelShuffle(up_scale)
            
        self.norm = AdaIN(d_out, d_cond) if d_cond is not None else nn.InstanceNorm1d(d_out, affine=False)
        
        
    def forward(self, x, cond=None):
        hid = self.conv(x.contiguous().transpose(1, 2))
        hid = self.dropout(hid)
        hid = F.gelu(hid)
        
        if self.up_scale != 1:
            hid = self.shuffle(hid)
            
        hid = hid.contiguous().transpose(1, 2)
        
        if isinstance(self.norm, AdaIN):
            out = self.norm(hid, cond)
        else:
            out = self.norm(hid)
        
        return out
        

        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_hid, d_head, dropout, return_attn=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_hid, d_head, dropout=dropout)
        self.return_weightAttn = return_attn

    def forward(self, query, key=None):
        """ (B, T, C) -> (B, T, C) """
        if key is None:
            key = query
        
        tot_timeStep = query.shape[1]       # (B, T, C)
        
        query = query.contiguous().transpose(0, 1)
        key = key.contiguous().transpose(0, 1)

        query, weight_Attn = self.attn(query, key, key)

        query = query.contiguous().transpose(0, 1)              # (B, T, C)

        if self.return_weightAttn:
            return query, weight_Attn
        else:
            return query
        
        
        
class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        sr = config['Dataset']['sr']
        self.n_fft = config['Dataset']['n_fft']
        self.hop_size = config['Dataset']['hop_size']
        dim_mel = config['Dataset']['dim_mel']

        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_size,
            center=False,
            power=1.0,
            norm="slaney",
            n_mels=dim_mel,
            mel_scale="slaney",
        )

    def forward(self, wav):
        wav = F.pad(wav, ((self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel
        
        
class Spectrogram(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        sr = config['Dataset']['sr']
        self.n_fft = config['Dataset']['n_fft']
        self.hop_size = config['Dataset']['hop_size']
        dim_mel = config['Dataset']['dim_mel']

        self.spctrogram = transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_size,
            center=False,
            power=1.0,
        )

    def forward(self, wav):
        wav = F.pad(wav, ((self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2), "reflect")
        spec = self.spctrogram(wav)
        return spec
        
        
        

""" SoftPlus """

class SoftPlus(nn.Module):
    def forward(self, input_tensor):
        return _softplus.apply(input_tensor)

class _softplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result=torch.log(1+torch.exp(i))
        ctx.save_for_backward(i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*torch.sigmoid(ctx.saved_variables[0])




""" Adaptive Instance Norm """

class AdaIN(nn.Module):
    def __init__(self, d_hid, d_cond):
        super().__init__()

        self.norm_layer = nn.InstanceNorm1d(d_hid, affine=False)
        self.linear = nn.Linear(d_cond, 2 * d_hid, bias=False)

    def forward(self, x, cond):
        """
        ? INPUT
            - x: (B, T, C)
            - cond: (B, 1, C)
        ? OUTPUT
            - (B, T, C), torch
        """
        if len(cond.shape) == 2:
            cond = cond.unsqueeze(1)
        
        scale, bias = self.linear(cond).chunk(2, dim=-1)
        return self.norm_layer(x) * scale + bias
        
def feature_norm(x, dim=1, eps: float = 1e-14):
    mean = torch.mean(x, dim=dim, keepdim=True)
    var = torch.square(x - mean).mean(dim=dim, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)




""" Pixel Shuffle """

class PixelShuffle(nn.Module):
    """ 
        Upsampling along time-axis + Downsampling along channel-axis.
    """
    def __init__(self, scale_factor: int):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """ 
        ? INPUT
        :x: tensor, (batch_size, channels, width)
        
        ? OUTPUT
        :return: tensor, (batch_size, out_channels, width),
            Shuffling pixels in the tensor to re-sample to suitable size,
            - channels = channels // scale_factor
            - width = width * scale_factor
        """ 
        batch_size, channels, in_width = x.size()
        channels = channels // self.scale_factor
        out_width = in_width * self.scale_factor
        
        x = x.contiguous().view(batch_size, channels, self.scale_factor, in_width)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, channels, out_width)
        return x



class InversePixelShuffle(nn.Module):
    """ 
        Downsampling along time-axis + Upsampling along channel-axis.
    """
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """ 
        ? INPUT
        :x: tensor, (batch_size, in_channels, width)
        
        ? OUTPUT
        :return: tensor, (batch_size, out_channels, width),
            Shuffling pixels in the tensor to re-sample to suitable size,
            - out_channels = in_channels * scale_factor
            - width = width // scale_factor
        """ 

        batch_size, in_channels, width = x.size()
        out_channels = in_channels * self.scale_factor
        width = width // self.scale_factor

        x = x.contiguous().view(batch_size, in_channels, width, self.scale_factor)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, out_channels, width)
        return x



class ConvBank(nn.Module):
    def __init__(self, c_in: int, c_out: int, n_bank = 8, bank_scale = 1):
        super(ConvBank, self).__init__()
        self.conv_bank = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d((k // 2, k // 2 - 1 + k % 2)),
                    nn.Conv1d(c_in, c_out, kernel_size=k),
                )
                for k in range(bank_scale, n_bank + 1, bank_scale)
            ]
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [self.act(layer(x)) for layer in self.conv_bank]
        out = torch.cat(outs + [x], dim=1)
        return out
