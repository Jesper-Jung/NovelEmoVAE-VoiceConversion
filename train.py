import os

import numpy as np
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
device = "cuda" if torch.cuda.is_available() else 'cpu'

from dataset import EmotionSpeechDataset, collate_fn
from torch.utils.data import DataLoader

#from model.textlesslib.textless.data.speech_encoder import SpeechEncoder
from model import EmotionStyleGenerationFlowVAE as ModelStructure

import fairseq
import wandb
import time
from tqdm import tqdm

from utils import makedirs, check_recon_mel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(0)

torch.manual_seed(44)


class Train():
    def __init__(self, config, config_preprocess):
        super().__init__()
        self.config = config
        self.wandb_login = config['Train']['wandb_login']
        self.sr = config['Dataset']['sr']


        """ HyperParameters """
        self.batch_size = config['Train']['batch_size']
        self.lr = config['Train']['learning_rate']
        lr_step = config['Train']['lr_step']
        
        self.num_workers = config['Train']['num_workers']
        
        self.beta_LL = config['Train']['beta_LL']
        self.beta_ent= config['Train']['beta_ent']
        
        self.annealing_init = config['Train']['annealing_initial_step']
        self.annealing_end = config['Train']['annealing_end_step']
        weight_decay = config['Train']['weight_decay']
        
        self.pitch_shift = config['Dataset']['use_pitch_shift']


        """ Model, Optimizer """
        
        self.mode_unit_discrete = config['Train']['mode_unit_discrete']
        self.model = ModelStructure(config).to(device)

        print("Whole Params: {}".format(self.get_n_params(self.model)))
        print("Encoder: {}".format(self.get_n_params(self.model.adain_encoder)))
        print("Decoder: {}".format(self.get_n_params(self.model.adain_decoder)))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_step)


        """ Dataset """
        self.hop_size = config['Dataset']['hop_size']
        self.croppedMel_len = config['Dataset']["length_mel"]
        self.croppedWav_len = config['Dataset']["length_mel"] * self.hop_size

        self.dataset_train = EmotionSpeechDataset(config, config_preprocess, 'train')
        self.dataset_eval = EmotionSpeechDataset(config, config_preprocess, 'test_u2u')

        """ Path """
        # Save Model Path: "./assets/ts/model/"
        # Save Fig: "./assets/ts/figs/"
        self.dir_path = config['Result']['asset_dir_path']
        self.save_step = config['Train']['save_for_step']
        


    def fit(self, tot_epoch):
        """ Training DataLoader """
        self.train_loader = DataLoader(
            self.dataset_train, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers
        )

        """ Evaluation DataLoader """
        self.eval_loader = DataLoader(
            self.dataset_eval, batch_size=self.batch_size, drop_last=True,
            shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers
        )

        self.cur_step = 0
        if args.saved_step > 0:
            self.cur_step = args.saved_step
            
            ###==== Load state dict
            pth_ckpt = args.saved_dir_path + '/checkpoint_{}.pth.tar'.format(args.saved_step)
            ckpt = torch.load(pth_ckpt)
            self.model.load_state_dict(ckpt['model'])
            
        
        self.outer_pbar = tqdm(total=tot_epoch, 
                               desc="Training... >>>> Total {} Epochs".format(tot_epoch), 
                               position=0)

        ts = time.time()
        self.asset_path = self.dir_path + str(ts)
        makedirs(self.asset_path)


        """ Make Funny Training!! """
        for epo in range(tot_epoch):
            self.training_step()
            self.validation_step()


    def step(self, batch):
        
        wav, unit, spk_emb, emo_id = list(map(lambda x: x.to(device), batch))
        if self.pitch_shift:
            wav = self.__pitch_shift(wav)
        
        
        # forward
        mel_recon_1, mel_recon_2, mel_recon_3, mel_true, loss = self.model(wav, unit, spk_emb, emo_id)
        
        # loss
        loss_mel_1, loss_mel_2, loss_mel_3, loss_flowLL, loss_H_post, loss_emo_pred = loss
        loss_total = loss_mel_1 + loss_mel_2 + loss_mel_3 + loss_emo_pred + self.beta_LL * loss_flowLL + self.beta_ent * loss_H_post #* self.annealing(self.cur_step, self.annealing_end, self.annealing_init)

        return loss_total, loss_mel_1, loss_mel_2, loss_mel_3, loss_flowLL, loss_H_post, loss_emo_pred


    def training_step(self):
        self.model.train()

        for batch in tqdm(self.train_loader):
            """ Training step """
            # zero grad
            self.optimizer.zero_grad()
            
            # forward & calculate total loss
            loss_total, loss_mel_1, loss_mel_2, loss_mel_3, loss_flowBPD, loss_H_post, loss_emo_pred  = self.step(batch)

            # backwarding
            loss_total.backward()

            # optimize
            self.optimizer.step()
            self.scheduler.step()


            """ end """
            loss_dict = {
                "Total Loss": loss_total.item(), 
                "Mel Recon Loss": loss_mel_3.item(),
                "Flow BPD Loss": loss_flowBPD.item(), 
                "Flow H Loss": loss_H_post.item(), 
                "Emo Cls Loss": loss_emo_pred.item(), 
            }
            
            self.training_step_end(loss_dict)      


    def training_step_end(self, loss_dict):
        self.cur_step += 1

        # Update current learning rate in the dictionary
        loss_dict.update( {"lr": self.optimizer.param_groups[0]['lr']} )
        self.outer_pbar.set_postfix(loss_dict)

        if self.wandb_login:
            loss_dict["Annealing"] = self.annealing(self.cur_step, self.annealing_end, self.annealing_init)
            wandb.log(loss_dict, step=self.cur_step)

        if self.cur_step % self.save_step == 0:
            save_path = os.path.join(self.asset_path, 'checkpoint_{}.pth.tar'.format(self.cur_step))
            save_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }

            torch.save(save_dict, save_path)
            print("save model at step {} ...".format(self.cur_step))


    def step_eval(self, batch):
        self.model.eval()
       
        wav, unit, spk_emb, emo_id = list(map(lambda x: x.to(device), batch))
        
        with torch.no_grad():
            mel_recon_1, mel_recon_2, mel_recon_3, mel_true, loss = self.model(wav, unit, spk_emb, emo_id)
            
        # loss
        loss_mel_1, loss_mel_2, loss_mel_3, loss_flowLL, loss_H_post, loss_emo_pred = loss
        loss_total = loss_mel_1 + loss_mel_2 + loss_mel_3 + loss_emo_pred + self.beta_LL * loss_flowLL + self.beta_ent * loss_H_post * self.annealing(self.cur_step, self.annealing_end, self.annealing_init)

        return mel_recon_1, mel_recon_2, mel_recon_3, mel_true, loss_total, loss_mel_1, loss_mel_2, loss_mel_3, loss_flowLL, loss_H_post, loss_emo_pred
    

    def validation_step(self):
        with torch.no_grad():
            eval_pbar = tqdm(self.eval_loader, desc="Validation...")

            for batch in eval_pbar:
                mel_recon_1, mel_recon_2, mel_recon_3, mel_true, loss_total, loss_mel_1, loss_mel_2, loss_mel_3, loss_flowBPD, loss_H_post, loss_emo_pred = self.step_eval(batch)

                """ log """
                loss_dict = {
                    #"Mel Post Loss": loss_post.item(),
                    "E Mel Recon Loss": loss_mel_3.item(),
                    "E Flow BPD Loss": loss_flowBPD.item(),
                    "E Flow H Loss": loss_H_post.item(),
                    "E Emo Cls Loss": loss_emo_pred.item(),
                }

                eval_pbar.set_postfix(loss_dict)

        if self.wandb_login:
            wandb.log(loss_dict)

        check_recon_mel(mel_recon_3[-1].to('cpu').detach().numpy(),      # (dim_mel, len_mel)
            self.asset_path, self.outer_pbar.n, mode='Recon')
        check_recon_mel(mel_true[-1].to('cpu').detach().numpy(),      # (dim_mel, len_mel)
            self.asset_path, self.outer_pbar.n, mode='GT')

        with open(self.asset_path + "/config.yaml", 'w') as file:
            documents = yaml.dump(self.config, file)

        self.outer_pbar.update()
        eval_pbar.close()


    def get_n_params(self, model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
    
    
    def get_unit_representation(self, wav):
        _s = time.time()
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2), "reflect")
        padding_mask = torch.BoolTensor(wav.shape).fill_(False)
        
        if self.mode_unit_discrete is False:
            inputs = {
                "source": wav.to(device),
                "padding_mask": padding_mask.to(device),
                "output_layer": 12,  # layer 12
            }
            
            with torch.no_grad():
                unit = self.unit_transformation.extract_features(**inputs)[0]
                
        else:
            # unit = torch.vstack(
            #     [self.unit_transformation(wav[i].to(device))['units'] for i in range(self.batch_size)]
            # ).long()
            unit = self.unit_transformation(wav.to(device).reshape(-1))['units'].reshape(self.batch_size, -1)
        _e = time.time()
        print(_e - _s)
        return unit
    
    def __pitch_shift(self, wav):
        cent = random.randint(-30, 30)
        wav = torchaudio.functional.pitch_shift(
            wav,
            sample_rate = 16000,
            n_steps = cent,
            bins_per_octave = 1200,
        )
        
        return wav
    
    
    def annealing(selt, step, ANNEALING_END_STEP, ANNEALING_INITIAL_STEP=20000):
        if step < ANNEALING_INITIAL_STEP:
            return 0.
        elif ANNEALING_INITIAL_STEP <= step and step < ANNEALING_END_STEP:
            return (step - ANNEALING_INITIAL_STEP) / (ANNEALING_END_STEP - ANNEALING_INITIAL_STEP)
        else:
            return 1.
        




import argparse
def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', type=int, 
        default=1000
    )
    parser.add_argument('--gpu_visible_devices', type=str, default='3, 4, 5')
    parser.add_argument('--saved_step', type=int, default=0)
    parser.add_argument('--saved_dir_path', type=str)

    args = parser.parse_args()

    return args




if __name__ == "__main__":
    import yaml
    config = yaml.load(
        open("./config/config.yaml", "r"), Loader=yaml.FullLoader
    )
    
    config_preprocess = yaml.load(
        open("./config/config_preprocess.yaml", "r"), Loader=yaml.FullLoader
    )

    wandb_login = config['Train']['wandb_login']
    
    lr = config['Train']['learning_rate']
    beta_LL = config['Train']['beta_LL']
    beta_ent= config['Train']['beta_ent']
    
    mode_unit_discrete = config['Train']['mode_unit_discrete']
    

    args = argument_parse()

    if wandb_login:
        wandb.login()
        wandb_name = "{}LL_{}ENT_wn".format(beta_LL, beta_ent)
        wandb.init(project='Recon_MSDegree_4', name=wandb_name)

    trainer = Train(config, config_preprocess)
    trainer.fit(args.epochs)