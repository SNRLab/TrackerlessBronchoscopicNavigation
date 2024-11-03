from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image
# from options_eval import MonodepthEvalOptions

import networks
from layers import disp_to_depth, transformation_from_parameters, disp_to_depth_no_scaling

import csv

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')
    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images',
                        default = os.path.join(file_dir, "data2"))

    parser.add_argument('--output_dir', type=str,
                        help='path to a test image or folder of images',
                        default = '/code/data2/Model_Results')
    
    parser.add_argument('--output_folderName', type=str,
                        help='path to a test image or folder of images',
                        default = 'output_folderName')
    
    parser.add_argument("--gauss_mask_threshold",
                            type=float,
                            help="number of dataloader workers",
                            default=0.7)
    
    parser.add_argument('--model_path', type=str,
                        default = 'SelfSupervised',
                        help='path to the test model'
                        )
    
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    parser.add_argument("--pose_model_type",
                        type=str,
                        help="normal or shared",
                        default="separate_resnet",
                        choices=["posecnn", "separate_resnet", "shared"])
    
    parser.add_argument("--enable_gauss_mask",
                        help="weighing the loss with gauss mask",
                        action="store_false")
    
    parser.add_argument("--frame_skip",
                            type=int,
                            help="sampling frequency while inference",
                            default=[3])
    
    
    return parser.parse_args()

def sample_filenames_frequency(filenames, sampling_frequency):
    outputfilenames = []
    count = 0
    outputfilenames.append(filenames[0])
    for file in filenames:
        
        if count == sampling_frequency:
            outputfilenames.append(file)
            count = 0 

        count+=1
    return outputfilenames

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def test_simple(args):
    
    """Function to predict for a single image or folder of images
    """
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = args.model_path

    pose_prediction = False
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    pose_ecoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")
    
    
    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False) 
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained 
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}

    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print(" Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict, strict=False)

    depth_decoder.to(device)
    depth_decoder.eval()
    
    # gauss 
    if args.enable_gauss_mask:
        models = {}
        models['decompose'] = networks.UNet(3, 3)
        
        models['sigma_combined'] = networks.FCN(output_size = 16) # 4 sigma and mu x 3 for 3 gaussians
        models['gaussian{}'.format(1)] = networks.GaussianLayer(192)
        
        models['decompose'].to(device)
        models['sigma_combined'].to(device)
        models['gaussian{}'.format(1)].to(device)
        
        # train gaussian 
        
        
        for n in ['decompose', 'gaussian1', 'sigma_combined']:
            print("Loading {} weights...".format(n))
            path = os.path.join(args.model_path, "{}.pth".format(n))
            model_dict = models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            models[n].load_state_dict(model_dict)
        
        models['decompose'].eval()
        models['sigma_combined'].eval()
        models['gaussian{}'.format(1)].eval()
    
    # # pose # loading
    pose_encoder = networks.ResnetEncoder(18,num_input_images=2, pretrained = 1)
    loaded_dict_pose = torch.load(pose_ecoder_path, map_location=device)
    pose_encoder.load_state_dict(loaded_dict_pose)
    pose_encoder.to(device)
    pose_encoder.eval()
    
    pose_encoder = networks.ResnetEncoder(18,num_input_images=2, pretrained = 1)
    loaded_dict_pose = torch.load(pose_ecoder_path, map_location=device)
    pose_encoder.load_state_dict(loaded_dict_pose)
    pose_encoder.to(device)
    pose_encoder.eval()
    
    pose_decoder_model = networks.PoseDecoder(num_ch_enc=pose_encoder.num_ch_enc, num_input_features = 1, num_frames_to_predict_for=2) 
    pose_decoder_dict = torch.load(pose_decoder_path, map_location=device)
    pose_decoder_model.load_state_dict(pose_decoder_dict)
    pose_decoder_model.to(device)
    pose_decoder_model.eval()
    
    
    prefixed = [filename for filename in os.listdir('/code/splits/endovis') if filename.endswith(("backward.txt", "forward.txt"))]
    
    # output folder names
    folder_names = glob.glob(os.path.join(args.output_dir, '**/model/Output/*'))
    
    output_names = {}
    for folder in folder_names:
        a = folder[-1]
        if(a >='0'and a <='9'):
            tokens = folder.split('/')
            phantom = tokens[4]
            phantom_seq  = tokens[-1]
            if not output_names.get(phantom, 0):
                output_names[phantom] = {}
            if not output_names[phantom].get(phantom_seq, 0):
                output_names[phantom][phantom_seq] = folder
            
        
    # new paths based on sampling_frequency 
    for file in prefixed:
        
        tokens1 = file.split('_')
        file_phantom    = tokens1[3]
        file_seq        = tokens1[4]
        fw              = tokens1[-1][:-4]
        
        output_directory = output_names[file_phantom][file_seq] + '/' + fw
        
        files_jpeg = glob.glob(os.path.join(output_directory + "**/*.jpeg"))
        # if not files_jpeg: 

        filenames_1 = readlines(os.path.join(os.path.dirname(__file__), "splits", "endovis",file)) 
        if not filenames_1:
            print("no files {}".format(os.path.join(os.path.dirname(__file__), "splits", "endovis",file))) 
            
        # if filenames_1:
        if filenames_1 and filenames_1[0].split('/')[5][:-4] == 'forward' : # and filenames_1[0].split('/')[2] != 'FullPhantom2'

            paths   = sample_filenames_frequency(filenames_1, sampling_frequency = args.frame_skip[0])[0:5]
            
            output_directory = output_directory + '/' + 'frequency_{}'.format(args.frame_skip[0])
            
            for i in range(len(paths)):
                if paths[i][0] == '/':
                    paths[i] = paths[i][1:]
            
            print("-> Predicting on {:d} test images".format(len(paths)))

            # PREDICTING ON EACH IMAGE IN TURN
            with torch.no_grad():
                for idx, image_path in enumerate(paths): # should be sequntial
                    
                    tokens_ = image_path.split('\t')
                    folder = tokens_[0]
                    f_str = "{:010d}.png".format(int(tokens_[1]))
                    image_path = os.path.join(args.image_path, folder, f_str)
                    

                    # Load image and preprocess
                    input_image = pil.open(image_path).convert('RGB')

                    original_width, original_height = input_image.size
                    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                    input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

                    gauss_mask_combined = []
                    if args.enable_gauss_mask:
                        # get a mask around images 
                        
                        # get the mask on two images and then multiply try on only one image
                        features      = models["decompose"](input_image) # no augmentation for validation 
                        decomposed    = features[1]
                        
                        sigma_out_combined        = models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
                        gaussian_mask1            = models["gaussian1"](sigma_out_combined[:, :4])
                        gaussian_mask2            = models["gaussian1"](sigma_out_combined[:, 4:8])
                        gaussian_mask3            = models["gaussian1"](sigma_out_combined[:, 8:12])
                        gaussian_mask4            = models["gaussian1"](sigma_out_combined[:, 12:16])
                        
                        gauss_mask_combined.append(gaussian_mask1[0]/4 + gaussian_mask2[0]/4 + gaussian_mask3[0]/4 + gaussian_mask4[0]/4)
                    
                        mask = torch.cat(gauss_mask_combined, 1) # for only one image
                        
                        mask[mask < args.gauss_mask_threshold] = 0
                        mask_t = torch.ones(input_image.shape).cuda()
                        mask_t[mask == 0] = 0
                        input_image=input_image*mask_t
                        
                        # save mask too
                        im = mask_t.squeeze().permute(1, 2, 0).cpu().numpy()
                        im = pil.fromarray(np.uint8(im*255))
                        output_name = os.path.splitext(os.path.basename(image_path))[0]
                        
                        # add frequency 
                        name_dest_im = os.path.join(output_directory, "mask_{}.jpeg".format(output_name))
                        im.save(name_dest_im, quality=95)
                    
                    # PREDICTION
                    # input_image = input_image.to(device)
                    features = encoder(input_image)
                    outputs = depth_decoder(features)

                   
                    disp = outputs[("disp", 0)]
                    disp_resized = disp
                    
                    # Saving numpy file
                    output_name = os.path.splitext(os.path.basename(image_path))[0]
                    name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                    # scaled_disp, depth_scaled = disp_to_depth(disp, 0.1, 150)
                    scaled_disp = disp_resized
                    depth_scaled = disp_to_depth_no_scaling(disp)
                    np.save(name_dest_npy, scaled_disp.cpu().numpy())
                    
                    # save depth 
                    name_depth_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                    np.save(name_depth_npy, depth_scaled.cpu().numpy())
                    

                    # Saving colormapped depth image
                    disp_resized_np = disp_resized.squeeze().cpu().numpy()
                    vmax = np.percentile(disp_resized_np, 95)

                    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax) # 归一化到0-1
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma') # colormap
                    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

                    im = pil.fromarray(colormapped_im)

                    name_dest_im = os.path.join(output_directory, "{}.jpeg".format(output_name))
                    im.save(name_dest_im, quality=95)

                    print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                        idx + 1, len(paths), name_dest_im))

                   
            print('->p Done!')
        # else:
        #     print('depth img exists in dir {}'.format(output_directory))

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
