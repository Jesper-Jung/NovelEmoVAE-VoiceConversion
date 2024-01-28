import torch
import torch.nn.functional as f
device = "cuda" if torch.cuda.is_available() else 'cpu'

import numpy as np
import os
import sys
import yaml
from tqdm import tqdm
import soundfile as sf

sys.path.append('./dataset')
from datasetLoader import EmotionSpeechDataset
from torchsummaryX import summary

from model import EmotionStyleGenerationFlowVAE as ModelStructure

config = yaml.load(
        open("./config/config.yaml", "r"), Loader=yaml.FullLoader
)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp



if __name__ == "__main__":
    model = ModelStructure(config).to(device)
    
    __wav_input = torch.zeros((1, 160*128)).float().to(device)
    __unit_input = torch.zeros((1, 64, 768)).float().to(device)
    __spk_emb = torch.zeros((1, 192)).float().to(device)
    __emo_id = torch.zeros((1)).long().to(device)
    __spec_input = torch.zeros((1, 128, 128)).float().to(device)
    
    with open("./paper/model_architecture.txt", 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"Layer Name: {name}, Parameter Shape: {param.size()}" + "\n")
    
    # summary(model, __wav_input, unit=__unit_input, spk_emb=__spk_emb, emo_id=__emo_id)
    result, params_info = summary_string(model.adain_encoder, __spec_input, emo_state=__emo_id)
    print(result)

    

    print("Whole Params: {}".format(get_n_params(model)))
    print("Encoder: {}".format(get_n_params(model.adain_encoder)))
    print("Decoder: {}".format(get_n_params(model.adain_decoder)))


