from __future__ import absolute_import, division, print_function

import os
# import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset2 import MonoDataset
from torchvision import transforms


class SCAREDDataset(MonoDataset):
    
    
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.resizeTransform = transforms.Resize((self.height, self.width),
                                        interpolation=self.interp)

    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip, resize = True):
        path = self.get_image_path(folder, frame_index, side)
        if not os.path.isfile(path):
            return print('no file{%s}'.format(path))
        color = self.loader(path)
        
        if resize: 
            color = self.resizeTransform(color.crop((0, 0, color.size[0], color.size[1]-64))) # defined in base class
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    def get_folder_path(self, filename):
        line = filename.split()
        folder = line[0]

        # implement a function to get data specific folder and frame_id
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
        return frame_index, folder, side


class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        depth_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


