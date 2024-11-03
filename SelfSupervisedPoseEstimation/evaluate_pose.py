from __future__ import absolute_import, division, print_function
from scipy.spatial.transform import Rotation as R
import os
import torch
import networks
import numpy as np
import math
from torch.utils.data import DataLoader
from layers import transformation_from_parameters_euler, transformation_from_parameters,euler_angles_to_matrix # transformation_from_parameters, 
from utils import readlines
from options_eval import MonodepthEvalOptions
from datasets import LungRAWDataset
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib
import glob

matplotlib.use('Agg') # non intercative for showing plots do: matplotlib.use('TkAgg',force=True)
# matplotlib.use('TkAgg',force=True)

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

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


def dump_r(source_to_target_transformations):
    rs = []
    cam_to_world = np.eye(4)
    rs.append(cam_to_world[:3, :3])
    for source_to_target_transformation in source_to_target_transformations:
        rs.append(cam_to_world[:3, :3])
    return rs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def compute_re(gtruth_r, pred_r):
    RE = 0
    gt = gtruth_r
    pred = pred_r
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose @ np.linalg.inv(pred_pose)
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return RE / gtruth_r.shape[0]

def compute_scale(gtruth, pred):

    # Optimize the scaling factor
    gtruth = gtruth[:len(pred), :, :]
    scale = np.sum(gtruth[:, :3, 3] * pred[:, :3, 3]) / np.sum(pred[:, :3, 3] ** 2)
    return scale


def euclideanDist(pt1, pt2):
    if len(pt1) != len(pt2):
        raise ValueError("Points must have the same number of dimensions")
    distance = math.sqrt(sum((coord1 - coord2) ** 2 for coord1, coord2 in zip(pt1, pt2)))
    return distance


def plotTrajectory(pred_poses, gt_local_poses, save_fig = False, name = 0):
    our_local_poses = pred_poses
    gt_local_poses = gt_local_poses[:len(pred_poses), :, :]
    dump_our = np.array(dump(our_local_poses))
    dump_gt = np.array(dump_gt_new(gt_local_poses))

    scale_our = dump_our * np.abs(compute_scale(dump_gt, dump_our))

    num = len(gt_local_poses) # shoudl be array
    points_our  = []
    points_gt   = []
    origin = np.array([[0], [0], [0], [1]])

    dist_pred = []
    dist_gt = []
    for i in range(0, num):
        point_gt = np.dot(dump_gt[i], origin)
        point_our = np.dot(scale_our[i], origin)

        dist_gt.append(euclideanDist(point_gt[:3], origin[:3]))
        dist_pred.append(euclideanDist(point_our[:3], origin[:3]))
        
        points_our.append(point_our)
        points_gt.append(point_gt)

    points_our  = np.array(points_our)
    points_gt   = np.array(points_gt)

    # new a figure and set it into 3d
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

    # set figure information
    # ax.set_title("3D_Curve")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")

    # draw the figure, the color is r = read
    figure1, = ax.plot(points_gt[:, 0, 0], points_gt[:, 1, 0], points_gt[:, 2, 0], c='b', linewidth=1.6)
    figure2, = ax.plot(points_our[:, 0, 0], points_our[:, 1, 0], points_our[:, 2, 0], c='g', linewidth=1.6)
    
    origin = [0, 0, 0]
    
    ax.quiver(*origin, 1, 0, 0, color='r', arrow_length_ratio=0.1)
    # Y-axis (green)
    ax.quiver(*origin, 0, 1, 0, color='g', arrow_length_ratio=0.1)
    # Z-axis (blue)
    ax.quiver(*origin, 0, 0, 1, color='b', arrow_length_ratio=0.1)

    if save_fig:
        plt.savefig('pose_prior_{}.png'.format(name),dpi=600)
    
    return plt


r = R.from_euler('z', 180, degrees=True).as_matrix()
k = np.eye(4)
k[:3,:3] = r
def dump(source_to_target_transformations):
    Ms = []
    cam_to_world = np.eye(4)
    Ms.append(cam_to_world)
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.matmul(cam_to_world, source_to_target_transformation)
        Ms.append(cam_to_world)
    return Ms

def get_transform(euler, translation):
    # the output of the network is in radians
    final_mat       = np.eye(4)
    final_mat[:3,:3]    = R.from_euler('zyx', euler.cpu().numpy().squeeze()).as_matrix()
    
    T = np.eye(4)
    T[:3,3] = translation.cpu().numpy().squeeze()
    M = np.matmul(T, final_mat)
    return M
    
def dump_gt_new(source_to_target_transformations):
    Ms = []
    cam_to_world = np.eye(4)
    Ms.append(cam_to_world)
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.matmul(cam_to_world, source_to_target_transformation) # consistent
      
        Ms.append(cam_to_world)
    return Ms

def evaluate(opt):
    """Evaluate odometry on the SCARED dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)


    prefixed = [filename for filename in os.listdir('/code/splits/endovis') if filename.endswith(("backward.txt", "forward.txt"))]
    
  
    folder_names = glob.glob(os.path.join(opt.log_dir, '**/SelfSupervised/Output/*'))

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
                
                
    for skip_frame_freq in opt.frame_skip:
        for file in prefixed:
        # for traj in range(1,5):
            
            tokens1 = file.split('_')
            file_phantom    = tokens1[3]
            file_seq        = tokens1[4]
            fw              = tokens1[-1][:-4]
            
            save_dir = output_names[file_phantom][file_seq] + '/' + fw
                
            filenames_1 = readlines(os.path.join(os.path.dirname(__file__), "splits", "endovis",file)) 
            if not filenames_1:
                print("no files {}".format(os.path.join(os.path.dirname(__file__), "splits", "endovis",file))) 
            
            
            if filenames_1 and filenames_1[0].split('/')[5][:-4] == 'forward':
                    
                filenames   = sample_filenames_frequency(filenames_1, sampling_frequency = skip_frame_freq)[1:-1]
                
                # optional 
                if not os.path.isdir(os.path.join(save_dir, 'frequency_{}'.format(skip_frame_freq))):
                    os.mkdir(os.path.join(save_dir, 'frequency_{}'.format(skip_frame_freq)))
                save_dir = save_dir + '/' + 'frequency_{}'.format(skip_frame_freq)
                
                
                # remoe first / 
                for i in range(len(filenames)):
                    if filenames[i][0] == '/':
                        filenames[i] = filenames[i][1:]
                        
                if fw == "backward":
                    if opt.flip_backward_images:
                        if not os.path.isdir(os.path.join(save_dir, 'flip')):
                            os.mkdir(os.path.join(save_dir, 'flip'))
                        save_dir = save_dir + '/' + 'flip'
                            
                    else:
                        if not os.path.isdir(os.path.join(save_dir, 'no_flip')):
                            os.mkdir(os.path.join(save_dir, 'no_flip'))
                        save_dir = save_dir + '/' + 'no_flip'

                if opt.flip_backward_images:
                    if fw == "forward":
                        continue
                
                dataset = LungRAWDataset(
                        opt.data_path, filenames, opt.height, opt.width,
                        [0, 1, -1], 4, is_train=False, len_ct_depth_data = len(filenames), data_augment = False, sampling_frequency = 1, depth_prior = False, random_frequency = False, flip_backward = opt.flip_backward_images)
                
                dataloader = DataLoader(dataset, 1, shuffle=False, drop_last=False, pin_memory=True)
                
               
                model_path = opt.load_weights_folder
                pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
                pose_decoder_path = os.path.join(model_path, "pose.pth")

                pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
                pose_encoder.load_state_dict(torch.load(pose_encoder_path))

                pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
                pose_decoder.load_state_dict(torch.load(pose_decoder_path))

                pose_encoder.cuda()
                pose_encoder.eval()
                pose_decoder.cuda()
                pose_decoder.eval()

                models = {}
                 
                if opt.enable_gauss_mask:
                   
                    
                    models['decompose'] = networks.UNet(3, 3)
                    
                    models['sigma_combined'] = networks.FCN(output_size = 16) # 4 sigma and mu x 3 for 3 gaussians
                    models['gaussian{}'.format(1)] = networks.GaussianLayer(192)
            
                    models['decompose'].to('cuda')
                    models['sigma_combined'].to('cuda')
                    models['gaussian{}'.format(1)].to('cuda')

                    # laod model
                    for n in opt.models_to_load:
                        print("Loading {} weights...".format(n))
                        path = os.path.join(model_path, "{}.pth".format(n))
                        model_dict = models[n].state_dict()
                        pretrained_dict = torch.load(path)
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        models[n].load_state_dict(model_dict)
                    
                    models['decompose'].eval()
                    models['sigma_combined'].eval()
                    models['gaussian{}'.format(1)].eval()
                   
                pred_poses = []

                print("-> Computing pose predictions")
                
                # write only if directory is not empty 
                count = 0 
                axisangle_ = []
                translation_ = []
                with torch.no_grad():
                    for inputs in dataloader:
                        
                        for key, ipt in inputs.items():
                            inputs[key] = ipt.cuda()
                        
                        frame1 = inputs[("color", 0, 0)]
                        frame2 = inputs[("color", 1, 0)]
                        
                        if opt.enable_gauss_mask:
                            gauss_mask_combined = []
                            for frame_id in [0, 1]:
                                features      = models["decompose"](inputs["color", frame_id, 0]) # no augmentation for validation 
                                decomposed    = features[1]
                                
                                sigma_out_combined        = models['sigma_combined'](features[0]) # will spit out 5, 1 gaussian std 
                                gaussian_mask1            = models["gaussian1"](sigma_out_combined[:, :4])
                                gaussian_mask2            = models["gaussian1"](sigma_out_combined[:, 4:8])
                                gaussian_mask3            = models["gaussian1"](sigma_out_combined[:, 8:12])
                                gaussian_mask4            = models["gaussian1"](sigma_out_combined[:, 12:16])
                                
                                gauss_mask_combined.append(gaussian_mask1[0]/4 + gaussian_mask2[0]/4 + gaussian_mask3[0]/4 + gaussian_mask4[0]/4)
                            
                            mask, idx = torch.min(torch.cat(gauss_mask_combined, 1), 1, keepdim = True) 
                            
                            mask[mask < opt.gauss_mask_threshold] = 0
                            # mask = mask[:, None, :, :]
                            mask_t = torch.ones(mask.shape).cuda()
                            mask_t[mask == 0] = 0
                            
                            frame1 = frame1*mask_t
                            frame2 = frame2*mask_t
                            
                        all_color_aug = torch.cat([frame1, frame2], 1)

                        features = [pose_encoder(all_color_aug)]
                        axisangle, translation = pose_decoder(features)
                        axisangle_.append(axisangle[:, 0].cpu().numpy())
                        translation_.append(translation[:, 0].cpu().numpy())

                        pred_poses.append(
                            transformation_from_parameters_euler(axisangle[:, 0], translation[:, 0]).cpu().numpy())
                        
                np.savez(os.path.join(save_dir, 'pose{}.npz'.format(1)), a = pred_poses)
                np.savez(os.path.join(save_dir,'eulerangle_{}.npz'.format(1)), a = axisangle_)
                np.savez(os.path.join(save_dir,'translation_{}.npz'.format(1)), a = translation_)
                
                
            else:
                print('directory_exists {}'.format(save_dir))
            
if __name__ == "__main__":
    options = MonodepthEvalOptions()
    evaluate(options.parse())
