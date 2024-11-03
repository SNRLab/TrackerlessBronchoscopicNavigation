

from __future__ import absolute_import, division, print_function

# from trainer import Trainer
from options import MonodepthOptions

import wandb_logging
import wandb
import numpy as np

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


from datasets import LungRAWDataset

from torchvision.utils import save_image, make_grid

import wandb_logging

import torchvision.transforms as transforms

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance

fid_criterion = FrechetInceptionDistance(feature = 64, normalize=True).to('cuda')


# losses 
def compute_reprojection_loss(pred, target, frac = 0.45):
    """Computes reprojection loss between a batch of predicted and target images
    """
    losses = {}
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1).mean(-1).mean(-1)
    

    ssim_loss = ssim(pred, target).mean(1).mean(-1).mean(-1)
    reprojection_loss = (1.0 - frac)* ssim_loss + frac * l1_loss
    
    losses['l1'] = l1_loss.mean(-1)

    losses['reprojection_loss'] = reprojection_loss.mean(-1) 
        
    losses['ssim_loss'] = ssim_loss.mean(-1)
    
    
        
    
    return losses

def save_model(epoch, log_path, models, model_optimizer):
    """Save model weights to disk
    """
    save_folder = os.path.join(log_path, "models", "weights_{}".format(epoch))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for model_name, model in models.items():
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = model.state_dict()
        if model_name == 'encoder':
            # save the sizes - these are needed at prediction time
            to_save['height'] = height
            to_save['width'] = width
        torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format("adam"))
    torch.save(model_optimizer.state_dict(), save_path)

def evaluation_FID(metric, imgs_dist1, imgs_dist2):
    metric.update(imgs_dist1, real=True)
    metric.update(imgs_dist2, real=False)
    return metric.compute()
    
    
    
def load_model_fxn(load_weights_folder, models_to_load, models):
    """Load model(s) from disk
    """
    load_weights_folder = os.path.expanduser(load_weights_folder)

    assert os.path.isdir(load_weights_folder), \
        "Cannot find folder {}".format(load_weights_folder)
    print("loading model from folder {}".format(load_weights_folder))

    for n in models_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(load_weights_folder, "{}.pth".format(n))
        model_dict = models[n].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[n].load_state_dict(model_dict)

    
frac = 0.65
file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
load_model = True 
# data loader
models = {}
input_size = ()
learning_rate = 10e-08 # original is 10e-06
batch_size = 4
num_epochs = 25
parameters_to_train = []
train_unet_only = False

data_path         = os.path.join(file_dir, "data")
height            = 192
width             = 192
frame_ids         = [0, -1, 1]
device            = torch.device("cuda")
n_channels        = 3
n_classes         = 3
scheduler_step_size = 8
bool_multi_gauss = True
separate_mean_std = True
batch_norm = True 
data_aug = True
gauss_number =  1

b1               = 0.5
b2               = 0.999

experiment_name = 'name_here'
# wandb 
config = dict(
    height = height,
    width = width,
    epochs=num_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    dataset=experiment_name,
    frame_ids = frame_ids,
    augmentation = data_aug,
    align_corner="True", 
    pretrained_unet_model = load_model, 
    multi_gauss = bool_multi_gauss, 
    same_gauss_assumption = False)
        
wandb.login()
wandb.init(project="gaussian_test", config= config, dir = 'data/logs', name = experiment_name)

if batch_norm: 
    models['decompose'] = networks.UNet(n_channels, n_classes) 
else:
    models['decompose'] = networks.UNet_instanceNorm(n_channels, n_classes)

if not train_unet_only: 
    
    if separate_mean_std: 
        models['sigma_combined'] = networks.FCN(output_size = 16) # 4 sigma and mu x 3 for 3 gaussians
        
        for g in range(1, gauss_number+1): 
            models['sigma{}'.format(g)] = networks.FCN(output_size = 10) # 4 for each of std x, std y, mean x , mean y, two for the mixture of gaussians 
            models['gaussian{}'.format(g)] = networks.GaussianLayer(height)
    
        
        
models['decompose'].to(device)
models['sigma_combined'].to(device)

# models['gaussian'].to(device)

if not train_unet_only:
    for g in range(1, gauss_number+1): 
        models['sigma{}'.format(g)].to(device)
        models['gaussian{}'.format(g)].to(device)

parameters_to_train += list(models["decompose"].parameters())
parameters_to_train += list(models['sigma_combined'].parameters())

for g in range(1, gauss_number+1): 
    parameters_to_train += list(models['gaussian{}'.format(g)].parameters())
    

# optimizer
model_optimizer = optim.Adam(parameters_to_train, learning_rate)
model_lr_scheduler = optim.lr_scheduler.StepLR(model_optimizer, scheduler_step_size, 0.1)

if load_model:
    load_model_fxn('/code/code/folder/models/weights_19', ["decompose"], models)
    

# dataloader 
datasets_dict = {"endovis": datasets.LungRAWDataset}
dataset = datasets_dict['endovis']

fpath = os.path.join(os.path.dirname(__file__), "splits", "endovis", "{}_files_phantom.txt")

train_filenames = readlines(fpath.format("train"))
val_filenames = readlines(fpath.format("val"))
img_ext = '.png'

num_train_samples = len(train_filenames)
num_total_steps = num_train_samples // batch_size * num_epochs
# data_augment = False
train_dataset =  dataset(
    data_path, train_filenames,  height,  width,
    frame_ids, 4, is_train=True, img_ext=img_ext, len_ct_depth_data = 2271, data_augment = data_aug)
train_loader = DataLoader(train_dataset, batch_size, shuffle = True, drop_last=True)
    
val_dataset = dataset(data_path, val_filenames,  height,  width,frame_ids, 4, is_train=False, img_ext=img_ext, len_ct_depth_data = 0,  data_augment = data_aug) 
val_loader = DataLoader(val_dataset,  batch_size, shuffle = True, drop_last=True)
val_iter = iter( val_loader)

models['decompose'].to(device)

if not train_unet_only:
    for g in range(1, gauss_number+1): 
        models['sigma{}'.format(g)].to(device)
        models['gaussian{}'.format(g)].to(device) 

# losses 
ssim = SSIM()
ssim.to(device)

# train loop
epoch = 0
step = 0
start_time = time.time()
step = 1
save_frequency = 20
custom_step = 0
prev_error = 100000000
prev_fid = 100000
for  epoch in range(num_epochs):
    print("Training")
    
    custom_step+=1
    outputs = {}
    
    for batch_idx, inputs in enumerate(train_loader):
        
        models['decompose'].train()
        models['sigma_combined'].train()
        
        if not train_unet_only:
            for g in range(1, gauss_number+1): 
                models['sigma{}'.format(g)].train()
                models['gaussian{}'.format(g)].train() 
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(device)
            
        before_op_time = time.time()

        total_loss = {'reprojection_loss':0, 'l1': 0, 'ssim_loss':0}

        total_fid = {'fid':0}
        
        
        #fcn model output mixture proportions too 
        for frame_id in [0, -1, 1]:
            # change to hsv 
            features                = models["decompose"](inputs["color_aug", frame_id, 0])
            outputs['decompose']    = features[1]
            
            
            sigma_out_combined        = models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
            gaussian_mask1            = models["gaussian1"](sigma_out_combined[:, :4])
            gaussian_mask2            = models["gaussian1"](sigma_out_combined[:, 4:8])
            gaussian_mask3            = models["gaussian1"](sigma_out_combined[:, 8:12])
            gaussian_mask4            = models["gaussian1"](sigma_out_combined[:, 12:16])
            
            outputs['compose'] = outputs['decompose'] * (gaussian_mask1[0]/4 + gaussian_mask2[0]/4 + gaussian_mask3[0]/4 + gaussian_mask4[0]/4)

            
            losses = compute_reprojection_loss(outputs['compose'], inputs["color_aug", frame_id, 0], frac)
            
            total_loss['l1']+=losses['l1']
            total_loss['reprojection_loss']+=losses['reprojection_loss']
            total_loss['ssim_loss']+=losses['ssim_loss']
            
            total_fid['fid']+=evaluation_FID(fid_criterion, inputs["color_aug", frame_id, 0], outputs['compose'])
        
        # for 3 frames
        total_loss['l1']/=3
        total_loss['reprojection_loss']=total_loss['reprojection_loss']/3
        total_loss['ssim_loss']/=3
        total_fid['fid']/=3
        
        model_optimizer.zero_grad()
        total_loss['reprojection_loss'].backward()
        model_optimizer.step()
        
        duration = time.time() - before_op_time
        
        step+=1
        
        # save model
        if total_fid['fid'] < prev_fid: 
            # save_model 
            save_model(epoch, 'code/{}'.format(experiment_name), models, model_optimizer)
            prev_fid = total_fid['fid']
        
        # wand_b loggin 
        if ( step + 1) %  save_frequency == 0:
            with torch.no_grad():
                models['decompose'].eval()
                models['sigma_combined'].eval()
                if not train_unet_only: 
                    models['gaussian1'].eval()
                    
                features_val        = models["decompose"](inputs["color_aug", 0, 0])
                image               = features_val[1]
                if not train_unet_only: 
                    sigma_out_val1                = models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
                    gaussian_mask_val1            = models["gaussian1"](sigma_out_combined[:, :4])
                    gaussian_mask_val2            = models["gaussian1"](sigma_out_combined[:, 4:8])
                    
                    gaussian_mask_val3            = models["gaussian1"](sigma_out_combined[:, 8:12])
                    gaussian_mask_val4            = models["gaussian1"](sigma_out_combined[:, 12:16])

                    
                    final_val = image * (gaussian_mask_val1[0]/4 + gaussian_mask_val2[0]/4 + gaussian_mask_val3[0]/4 + gaussian_mask_val4[0]/4)
                        
                
                wandb.log({"{}".format('train_original'):wandb.Image(inputs["color_aug", 0, 0], caption ='original image'),'custom_step':custom_step})  
                wandb.log({"{}".format('train_intermediate'):wandb.Image(make_grid(image), caption = 'intermediate image'),'custom_step':custom_step})  
                
                if not train_unet_only: 
                    wandb.log({"{}".format('train_reconstructed'):wandb.Image(make_grid(final_val), caption = 'reconstructed image'),'custom_step':custom_step})
                    wandb.log({"{}".format('train_gaussmask1'):wandb.Image(make_grid(gaussian_mask_val1[0]), caption = 'gaussian mask1'),'custom_step':custom_step})
                    wandb.log({"{}".format('train_gaussmask2'):wandb.Image(make_grid(gaussian_mask_val2[0]), caption = 'gaussian mask2'),'custom_step':custom_step}) 
                    wandb.log({"{}".format('train_gaussmask3'):wandb.Image(make_grid(gaussian_mask_val3[0]), caption = 'gaussian mask3'),'custom_step':custom_step}) 
                    wandb.log({"{}".format('train_gaussmask4'):wandb.Image(make_grid(gaussian_mask_val4[0]), caption = 'gaussian mask4'),'custom_step':custom_step})    
                    
                
                wandb.log({"{}".format('learning_rate'):model_lr_scheduler.optimizer.param_groups[0]['lr'],'custom_step':custom_step})
                for l, v in total_loss.items():
                    wandb.log({"{}_{}".format('train', l):v, 'custom_step':custom_step})
                
                # FID 
                for l, v in total_fid.items():
                    wandb.log({"{}_{}".format('train_fid', l):v, 'custom_step':custom_step})
                
                    
            models['sigma_combined'].train()
            models['decompose'].train()
            models['gaussian1'].train()
                
    model_lr_scheduler.step()
    #save model
    # save_model(epoch, 'code/logs', models, model_optimizer)
    
wandb.finish()
    
    
