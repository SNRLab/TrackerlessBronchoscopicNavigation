


import wandb
import torchvision
from torchvision import transforms
import torch

import PIL.Image as pil
from torchvision import transforms
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from typing import overload

class wandb_logging:
    
    def __init__(self, options, experiment_name = None, models = None):
        wandb.login()
        
        self.opts = options

        self.config = vars(options)
        self.config.update({'name':"phantom_Dataset_vanilla_hyperparameter_search"})
        self.config.update({'align_corner':"True"})
        self.config.update({'augmentation':"True"})
        self.config.update({'scaled_pose':"False"})
        self.config.update({'scaled_depth':"False"})
        self.config.update({'discLoss_depth_space':"True"})
        self.config.update({'corrected_color_aug_inLoss':"True"})
        self.config.update({'up_scaled_intermediate_depths_discriminator':"True"})
        
      
        self.resize = transforms.Resize((self.config['height'], self.config['width']))
        
        if experiment_name:
            wandb.init(project="all_phantom", config=self.config, dir = 'data/logs')
        else:
            wandb.init(project="all_phantom", config=self.config, dir = 'data/logs', name = experiment_name)
        
        self.save_colored_depth = False
        
        if models:
            self.models = []
            for model in models:
                self.models.append(model)
                wandb.watch(self.models[-1], log_freq=1000, log='all') # default is 1000, it makes the model very slow

        return 

    def startSweep(self, sweep_configuration, project_name, function_to_run, count):
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

        # wandb.agent(sweep_id, function=main, count=10)       
        wandb.agent(sweep_id, function=function_to_run, count=count)
        
    def finishWandb(self):
        wandb.finish()
        return
    
    def get_config(self):
        return self.config
        
    def log_data_stage2(self):
        return 
    
    def log_gradients(self, names=["gradient_norm"]):
        count = 0 
        for model in self.models:    
            # not correct throwing error 
            wandb.log({names[count]: model.named_parameters()})
            if len(names) > 1: 
                count+=1
        
    def log_lr(self, lr):
        wandb.log({'lr': lr})
    
    def log_data(self, outputs, losses, mode, step=1, character="registration", stage=1, learning_rate = 0, use_discriminator_loss = False, discriminator_loss = 0,
                 discriminator_response= None, gaussian_decomposition = False, gaussian_response = None):
       
        # log losses 
        # k = [key for key, value in losses.items()]
        
        for l, v in losses.items():
            wandb.log({"{}_{}".format(mode, l):v, 'custom_step':step})
        
        if use_discriminator_loss:
            wandb.log({"{}_disc_loss".format(mode):discriminator_loss, 'custom_step':step})
            
            
            
             
        wandb.log({"lr":learning_rate, 'custom_step':step})
        
        # log images 
        if outputs.get('trajectory', 0):
            wandb.log({"{}_{}".format(mode, 'trajectory'):wandb.Image(outputs['trajectory'], caption = ''),'custom_step':step})  


        # for j in range(min(4, self.config['batch_size'])):  # write a maxmimum of four images
        image_list_disc_res_ct = []
        if character != "trajectory":
            for s in self.config['scales']:
                image_list = []
                caption_list = []
                image_list_depth = []
                image_list_original = []
                image_list_automask = []
                image_list_color = []
                image_list_pred_color = []
                image_disc_response = []
                image_gauss_1 = []
                image_gauss_2 = []
                image_decompose = []
                image_compose = []
                
                for frame_id in self.config['frame_ids'][1:]: # what is logged here 

                # image_list.append(inputs[("color", 0, 0)][j].data)
                
                    # list_1 = outputs[("registration", s, frame_id)][:4,:,:]
                    if character=="registration":
                        image_list.append(outputs[("registration", s, frame_id)][:4,:,:])
                        
                if character=="disp":
                    
                    # colormap depth 
                    if self.save_colored_depth:
                        disp = outputs[("disp", s)][:1,:,:]
                        disp_resized_np = disp.squeeze().cpu().numpy()
                        vmax = np.percentile(disp_resized_np, 95)

                        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax) 
                        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma') # colormap
                        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

                        im = pil.fromarray(colormapped_im)
                        
                        image_list.append(im)
                        
                        wandb.log({"{}_{}".format(mode, s):wandb.Image(im, caption = ''),'custom_step':step})  
                    else:
                        image_list.append(outputs[("disp", s)][:4,:,:]) # first 4 images of the images
                        
                        if ("color_identity", 1, s) in outputs:
                            image_list_original.append(outputs[("color_identity", 1, s)][:4,:,:])
                        
                        if ("original_aug", 0, s) in outputs:
                            image_list_original.append(outputs[("original_aug", 0, s)][:4,:,:])
                            
                        if ("depth", 0, s) in outputs:
                            image_list_depth.append(outputs[("depth", 0, s)][:4,:,:])
                        
                        if "identity_selection/{}".format(s) in outputs:
                            image_list_automask.append(outputs["identity_selection/{}".format(s)][:4,:,:]*255)
                        
                        if ("color_aug_compose", frame_id, s) in outputs:
                            image_list_pred_color.append(outputs[("color_aug_compose", frame_id, s)][:4,:,:])
                            
                        if use_discriminator_loss:
                            # disc reponse 
                            if ("disc_response", s) in discriminator_response:
                                image_disc_response.append(discriminator_response[("disc_response", s)][:4,:,:])
                                
                      
                            
                if not self.save_colored_depth:
                    c = torch.concat(image_list, 0)
                    self.log_image_grid(mode = mode, image_list = c, scale = s, caption = ' ', character = ''.join((character,"{}".format(s))), step = step)
                    
                    if len(image_list_original) > 0:
                        c_original = torch.concat(image_list_original, 0)
                        img_grid = torchvision.utils.make_grid(c_original, normalize = True)
                        npimg = img_grid.permute(1, 2, 0).cpu().numpy()
                        self.log_single_image(''.join((mode,str(s),'original_aug_')), image = npimg, caption = "{}_{}_{}_{}".format(character, mode, s, ''.join('original_aug_')), step=step)
                        
                    
                    c_depth = torch.concat(image_list_depth, 0)
                    img_grid = torchvision.utils.make_grid(c_depth, normalize = True)
                    npimg = img_grid.permute(1, 2, 0).cpu().numpy()
                    self.log_single_image(''.join((mode,str(s),'depth_')), image = npimg, caption = "{}_{}_{}_{}".format(character, mode, s, ''.join('depth_')), step=step)
                    
                    # self.log_image_grid(mode = mode, image_list = c_depth, scale = s, caption = 'depth_', character = ''.join((character,"{}".format(s))), step = step)
                    if image_list_automask:
                        c_automask = torch.concat(image_list_automask, 0)
                        c_automask = c_automask[:, None, :, :]
                        self.log_image_grid(mode = mode, image_list = c_automask, scale = s, caption = 'automask_ ', character = ''.join((character,"{}".format(s))), step = step)
                    
                    if len(image_list_pred_color) > 0:
                        c_pred_color = torch.concat(image_list_pred_color, 0)
                        self.log_image_grid(mode = mode, image_list = c_pred_color, scale = s, caption = 'pred_color ', character = ''.join((character,"{}".format(s))), step = step)

                    if use_discriminator_loss:
                        c_discriminator_response = torch.concat(image_disc_response, 0)
                        img_grid_disc_res = torchvision.utils.make_grid(c_discriminator_response, normalize = True)
                        npimg_disc = img_grid_disc_res.permute(1, 2, 0).cpu().numpy()
                        self.log_single_image(''.join((mode,str(s),'disc_response_')), image = npimg_disc, caption = "{}_{}_{}_{}".format(character, mode, s, ''.join('disc_response_')), step=step)
                        
                    if gaussian_decomposition:
                         
                        if 'gaussian_mask1' in gaussian_response:
                            c_discriminator_response = torch.concat(gaussian_response['gaussian_mask1'], 2)
                            img_grid_disc_res = torchvision.utils.make_grid(c_discriminator_response, normalize = True)
                            npimg_disc = img_grid_disc_res.permute(1, 2, 0).cpu().numpy()
                            self.log_single_image(''.join((mode,str(s),'gauss1_response_')), image = npimg_disc, caption = "{}_{}_{}_{}".format(character, mode, s, ''.join('gauss1_response_')), step=step)
                            
                        if 'gaussian_mask2' in gaussian_response:
                            c_discriminator_response = torch.concat(gaussian_response['gaussian_mask2'], 2)
                            img_grid_disc_res = torchvision.utils.make_grid(c_discriminator_response, normalize = True)
                            npimg_disc = img_grid_disc_res.permute(1, 2, 0).cpu().numpy()
                            self.log_single_image(''.join((mode,str(s),'gauss2_response_')), image = npimg_disc, caption = "{}_{}_{}_{}".format(character, mode, s, ''.join('gauss2_response_')), step=step)
                        
                        if 'decomposed' in gaussian_response:
                            c_discriminator_response = torch.concat(gaussian_response['decomposed'], 2)
                            img_grid_disc_res = torchvision.utils.make_grid(c_discriminator_response, normalize = True)
                            npimg_disc = img_grid_disc_res.permute(1, 2, 0).cpu().numpy()
                            self.log_single_image(''.join((mode,str(s),'decompose_gauss_')), image = npimg_disc, caption = "{}_{}_{}_{}".format(character, mode, s, ''.join('decompose_gauss_')), step=step)
                            
                       
                        if 'original' in gaussian_response and len(gaussian_response['original']) > 0:
                            c_discriminator_response = torch.concat(gaussian_response['original'], 2)
                            img_grid_disc_res = torchvision.utils.make_grid(c_discriminator_response, normalize = True)
                            npimg_disc = img_grid_disc_res.permute(1, 2, 0).cpu().numpy()
                            self.log_single_image(''.join((mode,str(s),'original_')), image = npimg_disc, caption = "{}_{}_{}_{}".format(character, mode, s, ''.join('original_')), step=step)
               
            if use_discriminator_loss:
                image_list_disc_res_ct.append(discriminator_response[('disc_response_ct')][:4,:,:])
                ct_discriminator_response = torch.concat(image_list_disc_res_ct, 0)
                img_grid_disc_res_ct = torchvision.utils.make_grid(ct_discriminator_response, normalize = True)
                npimg_disc_ct = img_grid_disc_res_ct.permute(1, 2, 0).cpu().numpy()
                self.log_single_image(''.join((mode,str(s),'disc_response_ct_')), image = npimg_disc_ct, caption = "{}_{}_{}_{}".format(character, mode, s, ''.join('disc_response_ct_')), step=step)
    
    def save_model(self, path):
        return 
    
    def log_image_grid(self, mode, image_list, scale, caption, character = '', step= 1):
        img_grid = torchvision.utils.make_grid(image_list)
        
        npimg = img_grid.permute(1, 2, 0).cpu().numpy()
        self.log_single_image(''.join((mode,str(scale),caption)), image = npimg, caption = "{}_{}_{}_{}".format(character, mode, scale, ''.join(caption)), step=step)
        return 
    
    def log_single_image(self, dict_key, step, image, caption=''):
        wandb.log({"{}".format(dict_key):wandb.Image(image, caption = caption),'custom_step':step})  
        return 
    
    def log_single_image_fromtensor(self, dict_key, image, caption=''):
        
        npimg = image.cpu().numpy()
        if image[0]==3:
            npimg = image.permute(1, 2, 0).cpu().numpy()
            
        wandb.log({"{}".format(dict_key):wandb.Image(npimg, caption = caption)})  
        return 

    

        