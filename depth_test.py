from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import cv2

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth, transformation_from_parameters, BackprojectDepth, Project3D
from utils import download_model_if_doesnt_exist

RECONSTRUCTION = False

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        default="mono+stereo_640x192",
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_depth_pose(args):
    """Function to predict depth and pose
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained depth encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    print("   Loading pretrained pose encoder")
    pose_encoder = networks.ResnetEncoder(18, False, 2)
    loaded_dict_pose_enc = torch.load(pose_encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    pose_encoder.load_state_dict(loaded_dict_pose_enc)

    encoder.to(device)
    pose_encoder.to(device)
    encoder.eval()
    pose_encoder.eval()

    print("   Loading pretrained depth decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    print("   Loading pretrained pose decoder")
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    loaded_dict_pose = torch.load(pose_decoder_path, map_location=device)
    pose_decoder.load_state_dict(loaded_dict_pose)

    depth_decoder.to(device)
    pose_decoder.to(device)
    depth_decoder.eval()
    pose_decoder.eval()

    print("-> Predicting on test images")

    pred_depths = []
    pred_poses = []

    backproject_depth = BackprojectDepth(1, feed_height, feed_width)
    backproject_depth.to(device)
    project_3d = Project3D(1, feed_height, feed_width)
    project_3d.to(device)

    K = np.array([[0.58, 0, 0.5, 0],
                  [0, 1.92, 0.5, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    K[0, :] *= feed_width
    K[1, :] *= feed_height
    inv_K = np.linalg.pinv(K)

    K = torch.from_numpy(K)
    K = K.unsqueeze(0).to(device)
    inv_K = torch.from_numpy(inv_K)
    inv_K = inv_K.unsqueeze(0).to(device)


    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():

        for i in range(107):

            # Load image and preprocess
            image_0_path = './kitti_data/01/{:010d}.jpg'.format(i)
            input_image_0 = Image.open(image_0_path).convert('RGB')
            original_width, original_height = input_image_0.size
            input_image_0 = input_image_0.resize((feed_width, feed_height), Image.LANCZOS)
            input_image_0 = transforms.ToTensor()(input_image_0).unsqueeze(0)

            image_1_path = './kitti_data/01/{:010d}.jpg'.format(i + 1)
            input_image_1 = Image.open(image_1_path).convert('RGB')
            input_image_1 = input_image_1.resize((feed_width, feed_height), Image.LANCZOS)
            input_image_1 = transforms.ToTensor()(input_image_1).unsqueeze(0)

            # PREDICTION for depth
            input_image_0 = input_image_0.to(device)
            features = encoder(input_image_0)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            #disp_resized = torch.nn.functional.interpolate(
            #    disp, (original_height, original_width), mode="bilinear", align_corners=False)

            _, pred_depth = disp_to_depth(disp, 0.1, 100)
            pred_depth = pred_depth.cpu()[:, 0].numpy()

            pred_depths.append(pred_depth[0])

            print("   Predict Depth {:d}".format(i))

            # PREDICTION for pose
            input_image_1 = input_image_1.to(device)
            input_image_pose = torch.cat([input_image_0, input_image_1], 1)
            features_pose = pose_encoder(input_image_pose)
            features_pose = [features_pose]
            axisangle, translation = pose_decoder(features_pose)

            pred_pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0])

            pred_poses.append(pred_pose.cpu()[0].numpy())

            print("   Predict Pose {:d}".format(i))
            print(pred_pose)

            # WARPED image
            if RECONSTRUCTION:
                print("   Reconstruct image {:d}".format(i))
                cam_points = backproject_depth(pred_depth, inv_K)
                pix_coords = project_3d(cam_points, K, pred_pose)
                reconstruct_image_0 = torch.nn.functional.grid_sample(
                    input_image_1,
                    pix_coords,
                    padding_mode="border")
                print("   Saving resonstructed image...")

                reconstruct_image_0 = torch.nn.functional.interpolate(reconstruct_image_0,
                                                                      (original_height, original_width), mode="bilinear",
                                                                      align_corners=False)
                reconstruct_image_0_np = reconstruct_image_0.squeeze().cpu().numpy()
                reconstruct_image_0_np = (reconstruct_image_0_np * 255).astype(np.uint8)
                reconstruct_image_0_np = np.concatenate(
                    [np.expand_dims(reconstruct_image_0_np[i], 2) for i in range(3)], 2)
                im = Image.fromarray(reconstruct_image_0_np, mode='RGB')
                name_dest_im = os.path.join("kitti_data/01", "warped", "{:010d}_warped.jpg".format(i))
                im.save(name_dest_im)
            print("...")

    np.save('kitti_data/pred_depth_01.npy', np.array(pred_depths))
    np.save('kitti_data/pred_pose_01.npy', np.array(pred_poses))
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_depth_pose(args)


'''def evaluate():

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # get ground truth data
    gt_path = "kitti_data/01/gt_depths.npz"
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    ratios = []

    for i in range(107):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        # get prediction
        pred_disp = np.load('./kitti_data/01/{:010d}_disp.npy'.format(i), encoding="latin1")
        pred_disp = np.squeeze(pred_disp)
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        # define a mask for comparing pred with gt
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # compute ratios
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    print(ratios)


if __name__ == "__main__":
    evaluate()
'''