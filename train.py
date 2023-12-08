import sys
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else 'cpu'

from dataset import Dataset_ESD, Dataset_EmovDB, collate_fn
from torch.utils.data import DataLoader

from model import EmotionStyleGenerationFlowVAE as ModelStructure

import fairseq
import wandb
import time
from tqdm import tqdm

from utils import makedirs, check_recon_mel



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

torch.manual_seed(44)


contentVec_ckpt_path = "./model/contextVec/checkpoint_best_legacy_500.pt"


class Train():
    def __init__(self, config, list_configDataset):
        super().__init__()
        self.config = config
        self.wandb_login = config['Train']['wandb_login']
        self.sr = config['Dataset']['sr']


        """ HyperParameters """
        self.batch_size = config['Train']['batch_size']
        self.lr = config['Train']['learning_rate']
        
        self.num_workers = config['Train']['num_workers']
        self.beta_LL = config['Train']['beta_LL']
        weight_decay = config['Train']['weight_decay']


        """ Model, Optimizer """
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([contentVec_ckpt_path])
        self.unit_transformation = models[0]            # Context Vec Model
        self.unit_transformation.to(device)
        self.unit_transformation.eval()
        
        self.model = ModelStructure(config).to(device)

        print("Autoencoder: {}".format(self.get_n_params(self.model)))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)


        """ Dataset """
        self.hop_size = config['Loader']['hop_size']
        self.croppedMel_len = config['Loader']["length_mel"]
        self.croppedWav_len = config['Loader']["length_mel"] * self.hop_size
        
        config_ESD, config_EmovDB = list_configDataset

        self.dataset_train = Dataset_ESD(config_ESD, 'train') + Dataset_EmovDB(config_EmovDB, 'train')
        self.dataset_eval = Dataset_ESD(config_ESD, 'eval') + Dataset_EmovDB(config_EmovDB, 'eval')

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
        wav, wav_aug, spk_emb, emo_id = list(map(lambda x: x.to(device), batch))
        
        # transformation (train)
        unit = self.get_unit_representation(wav_aug)
        
        # forward
        mel_post, mel_recon, mel_true, loss = self.model(wav, unit, spk_emb, emo_id)
        
        # loss
        loss_post, loss_mel, loss_flowLL = loss
        loss_total = loss_post + loss_mel + self.beta_LL * loss_flowLL

        return loss_total, loss_post, loss_mel, loss_flowLL 


    def training_step(self):
        self.model.train()

        for batch in tqdm(self.train_loader):
            """ Training step """
            # zero grad
            self.optimizer.zero_grad()
            
            # forward & calculate total loss
            loss_total, loss_post, loss_mel, loss_flowBPD  = self.step(batch)

            # backwarding
            loss_total.backward()

            # optimize
            self.optimizer.step()


            """ end """
            loss_dict = {
                "Total Loss": loss_total.item(), 
                "Post Loss": loss_post.item(), 
                "Mel Loss": loss_mel.item(),
                "Flow BPD Loss": loss_flowBPD.item(), 
            }
            
            self.training_step_end(loss_dict)      


    def training_step_end(self, loss_dict):
        self.cur_step += 1

        # Update current learning rate in the dictionary
        loss_dict.update( {"lr": self.optimizer.param_groups[0]['lr']} )
        self.outer_pbar.set_postfix(loss_dict)

        if self.wandb_login:
            wandb.log(loss_dict)

        if self.cur_step % self.save_step == 0:
            save_path = os.path.join(self.asset_path, 'checkpoint_{}.pth.tar'.format(self.cur_step))
            save_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            torch.save(save_dict, save_path)
            print("save model at step {} ...".format(self.cur_step))


    def step_eval(self, batch):
        self.model.eval()
        
        wav, _, spk_emb, emo_id = list(map(lambda x: x.to(device), batch))
        
        # transformation (eval)
        unit = self.get_unit_representation(wav)
        
        with torch.no_grad():
            mel_post, mel_recon, mel_true, loss = self.model(wav, unit, spk_emb, emo_id)
            
        # loss
        loss_post, loss_mel, loss_flowBPD = loss
        loss_total = loss_post + loss_mel + self.beta_LL * loss_flowBPD

        return mel_post, mel_recon, mel_true, loss_total, loss_post, loss_mel, loss_flowBPD
    

    def validation_step(self):
        with torch.no_grad():
            eval_pbar = tqdm(self.eval_loader, desc="Validation...")

            for batch in eval_pbar:
                mel_post, mel_recon, mel_true, _, loss_post, loss_mel, loss_flowBPD = self.step_eval(batch)

                """ log """
                loss_dict = {
                    "Post Val Loss": loss_post.item(),
                    "Flow BPD Loss": loss_flowBPD.item()
                }

                eval_pbar.set_postfix(loss_dict)

        if self.wandb_login:
            wandb.log(loss_dict)

        check_recon_mel(mel_post[-1].to('cpu').detach().numpy(),      # (dim_mel, len_mel)
            self.asset_path, self.outer_pbar.n, mode='recon')
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
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2), "reflect")
        padding_mask = torch.BoolTensor(wav.shape).fill_(False)
        
        inputs = {
            "source": wav.to(wav.device),
            "padding_mask": padding_mask.to(wav.device),
            "output_layer": 12,  # layer 12
        }
        
        with torch.no_grad():
            unit = self.unit_transformation.extract_features(**inputs)[0]
        
        return unit
        


        

import argparse
def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs', type=int, 
        default=800
    )
    parser.add_argument('--gpu_visible_devices', type=str, default='3, 4, 5')

    args = parser.parse_args()

    return args




if __name__ == "__main__":
    import yaml
    config = yaml.load(
        open("./config/config.yaml", "r"), Loader=yaml.FullLoader
    )
    
    config_ESD = yaml.load(
        open("./config/ESD_dataset.yaml", "r"), Loader=yaml.FullLoader
    )
    
    config_EmovDB = yaml.load(
        open("./config/EmovDB_dataset.yaml", "r"), Loader=yaml.FullLoader
    )

    wandb_login = config['Train']['wandb_login']
    lr = config['Train']['learning_rate']

    args = argument_parse()

    if wandb_login:
        wandb.login()
        wandb_name = "Recon_VC"
        wandb.init(project='Recon_Emotion_VC', name=wandb_name)

    trainer = Train(config, [config_ESD, config_EmovDB])
    trainer.fit(args.epochs)