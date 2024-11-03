from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES=True

def pil_loaded_gray(path):
    with open(path, 'rb') as f:
        with Image.open(f).convert('L') as img:
            return img
            
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 len_ct_depth_data, 
                 is_train=False,
                 img_ext='.png', 
                 sampling_frequency = 2, 
                 data_augment = True, 
                 depth_prior = False, 
                 random_frequency = True, 
                 flip_backward = True
                 ):
        super(MonoDataset, self).__init__()

        self.len_ct_depth_data = len_ct_depth_data
        
        self.data_aug   = data_augment
        self.data_path  = data_path
        self.filenames  = filenames
        self.height     = height
        self.width      = width
        self.num_scales = num_scales
        self.interp     = Image.ANTIALIAS
        self.random_frequency       = random_frequency
        self.frame_idxs             = frame_idxs
        self.flip_backward          = flip_backward

        self.is_train = is_train
        self.img_ext = img_ext
        self.depth_prior_endoslam = depth_prior
        self.loader = pil_loader
        self.loader_gray = pil_loaded_gray
        self.to_tensor = transforms.ToTensor()
        self.to_PIL = transforms.ToPILImage()
        
        self.sampling_frequency = sampling_frequency
        
        self.resize_ct = transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR)
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)

        except TypeError:
            self.brightness = 0.5
            self.contrast = 0.3
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                         interpolation=self.interp)

        self.transforms_aug = transforms.Compose([
                transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue),
                transforms.RandomHorizontalFlip(p=0.5),
               
                ])
        
        
        
        self.load_depth = False

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                if not inputs.get((n + "_aug", im, i), 0):
                    inputs[(n + "_aug", im, i)] = 0 
                    
                # save_image(self.to_tensor((f)), 'no_aug_{}_{}.png'.format(im, i))
                state = torch.get_rng_state()
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                # save_image(inputs[(n + "_aug", im, i)], 'after_aug_{}_{}.png'.format(im, i))
                torch.set_rng_state(state)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        
        # sample random number between 1 to the sampling frequency inclusive. 
        if self.random_frequency:
            curr_sampling_rate = np.random.randint(1, self.sampling_frequency, size=1)[0]
        else:
            curr_sampling_rate = self.sampling_frequency 
            
        inputs = {}
   
        if self.data_aug: 
            do_color_aug = self.is_train and random.random() > 0.5
        else:
            do_color_aug = False
            
        
        do_flip = self.is_train and random.random() > 0.5

        frame_index, folder, side, fw = self.get_folder_path(self.filenames[index]) # incorrect
        
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                if fw == 'forward':
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index + i + i*(curr_sampling_rate - 1), side, do_flip, resize = True)
                elif fw == 'backward':
                     # flipping the frames
                    if self.flip_backward:
                        inputs[("color", (-1)*i, -1)] = self.get_color(folder, frame_index + i + i*(curr_sampling_rate - 1), side, do_flip, resize = True)
                    else:
                        inputs[("color", i, -1)] = self.get_color(folder, frame_index + i + i*(curr_sampling_rate - 1), side, do_flip, resize = True)
                        
                else:
                    inputs[("color", (-1)*i, -1)] = self.get_color(folder, frame_index + i + i*(curr_sampling_rate - 1), side, do_flip, resize = True)
                    

        if self.depth_prior_endoslam:
            inputs[("color_depth", 0, 0)] = self.to_tensor(self.get_depth_endoslam(folder, frame_index))
        
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)


            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = self.transforms_aug
           
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
            
        return inputs

    def save_images(self, inputs, frame_index, folder="log_dataset_debug"):
        
        depth_3d = torch.cat([inputs[('color_aug', -1, 0)], inputs[('color_aug', 0, 0)], inputs[('color_aug', 1, 0)], inputs[('color_aug', 0, 0)]-inputs[('color_aug', -1, 0)], inputs[('color_aug', 1, 0)]-inputs[('color_aug', 0, 0)], inputs[('color_aug', 1, 0)]-inputs[('color_aug', -1, 0)]], 2)
        img = self.to_PIL(depth_3d)
        img.save(os.path.join(folder, "{}_check.jpg".format(str(frame_index)))) 
        return
    
    def get_folder_path(self, filename):
        raise NotImplementedError
    
    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    def get_depth_endoslam(self, folder, frame_index):
        raise NotImplementedError
    
    def get_depth_image_path(self, folder, frame_index):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

        
    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
