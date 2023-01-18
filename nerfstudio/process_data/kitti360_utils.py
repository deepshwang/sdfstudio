# adapted from
# https://github.com/EPFL-VILAB/omnidata
# https://github.com/autonomousvision/monosdf/preprocess/extract_monocular_cues.py
import argparse
import os
import os.path
import shutil
import sys
from glob import glob
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from PIL import Image
from torchvision import transforms

from nerfstudio.cameras.kitti360camera import KITTI360CameraPerspective
from nerfstudio.process_data.colmap_utils import rotmat2qvec
from omnidata.omnidata_tools.torch.data.transforms import get_transform
from omnidata.omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel
from omnidata.omnidata_tools.torch.modules.unet import UNet


def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths


def save_outputs(img_path, save_path, model, task, device):
    image_size = 384  # omnidata model input size
    w = 1408
    h = 376
    if task == "depth":
        trans_totensor = SplitToTensor(w, h, image_size)
    else:
        trans_totensor = SplitToTensor(w, h, image_size, normalize=False, g_transform=True, totensor=False)
    trans_topil = transforms.ToPILImage()
    with torch.no_grad():
        print(f"Reading input {img_path} ...")
        img = Image.open(img_path)

        img_tensors = trans_totensor(img)
        img_tensors = [t[:3].unsqueeze(0).to(device) for t in img_tensors]
        # img_tensors = trans_totensor(img)[:3].unsqueeze(0).to(device)

        if task == "depth":
            depth_concat = torch.zeros((376, 1408))
        else:
            depth_concat = torch.zeros(3, 376, 1408)
        outputs = []
        cum_ratio = 1
        for j, img_tensor in enumerate(img_tensors):
            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3, 1)

            output = model(img_tensor).clamp(min=0, max=1)
            if task == "depth":
                output = output.unsqueeze(0)
            output = F.interpolate(output, (376, 376))
            if task == "depth":
                # output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
                output = output.clamp(0, 1)
                outputs.append(output)
                if j < len(img_tensors) - 1:
                    if j != 0:
                        ratio = torch.median(outputs[j - 1][:, :, :, -1] / output[:, :, :, 0]).detach().cpu()
                        cum_ratio *= ratio
                    depth_concat[:, 376 * j : 376 * (j + 1)] = cum_ratio * output.detach().cpu().squeeze()
                else:
                    rem = 1408 % 376
                    oo = output[:, :, :, -rem:]
                    ratio = torch.median(outputs[j - 1][:, :, :, -1] / oo[:, :, :, 0]).detach().cpu()
                    cum_ratio *= ratio
                    depth_concat[:, 376 * j :] = cum_ratio * oo.detach().cpu().squeeze()
                # np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])

                # output = 1 - output
            #             output = standardize_depth_map(output)
            # plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')

            else:
                # output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
                outputs.append(output)
                if j < len(img_tensors) - 1:
                    if j != 0:
                        ratio = torch.median(outputs[j - 1][:, :, :, -1] / output[:, :, :, 0]).detach().cpu()
                        cum_ratio *= ratio
                    depth_concat[:, :, 376 * j : 376 * (j + 1)] = output.detach().cpu().squeeze()

                else:
                    rem = 1408 % 376
                    oo = output[:, :, :, -rem:]
                    ratio = torch.median(outputs[j - 1][:, :, :, -1] / oo[:, :, :, 0]).detach().cpu()
                    cum_ratio *= ratio
                    if task == "depth":
                        depth_concat[:, 376 * j :] = cum_ratio * oo.detach().cpu().squeeze()
                    else:
                        depth_concat[:, :, 376 * j :] = oo.detach().cpu().squeeze()
                # np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])
                # trans_topil(output[0]).save(save_path)
        if task == "depth":
            plt.imsave(save_path, depth_concat.detach().cpu().squeeze(), cmap="viridis")
        else:
            trans_topil(depth_concat).save(save_path)
        np.save(str(save_path).replace(".png", ".npy"), depth_concat.detach().cpu().numpy())
        print(f"Writing output {save_path} ...")


class SplitToTensor:
    def __init__(self, w, h, image_size, normalize=True, g_transform=False, totensor=True):
        # resizing parameters
        self.image_size = image_size

        self.w = w
        self.h = h
        self.n_splits = w // h + 1
        # normalization parameters
        self.normalize = normalize
        self.mean = 0.5
        self.std = 0.5
        self.inplace = False
        self.g_transform = None
        if g_transform:
            self.g_transform = get_transform("rgb", image_size=None)
        self.totensor = totensor

    def __call__(self, pic):
        outs = []
        for s in range(self.n_splits):
            if s < (self.n_splits - 1):
                out = VF.crop(pic, 0, self.h * s, self.h, self.image_size)
            else:
                out = VF.crop(pic, 0, self.w - self.h, self.h, self.h)
            out = VF.resize(out, (self.image_size, self.image_size), PIL.Image.BILINEAR, None, None)
            if self.g_transform is not None:
                out = self.g_transform(out)
            if self.totensor:
                out = VF.to_tensor(out)
            if self.normalize:
                out = VF.normalize(out, self.mean, self.std, self.inplace)
            outs.append(out)
        return outs


def split_and_save(data, save_data, seq_id, stereo_id, start_img_id, end_img_id, extract_priors=True):
    # [1] Prepare auxiliaries
    omnidata_weight_dir = "./omnidata/omnidata_tools/torch/pretrained_models/"
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_filenames = []
    cam2worlds = []
    fx = []
    fy = []
    cx = []
    cy = []
    stereo_ids = []
    Ks = []

    # [2] Load KITTI-360 camera of the selected sequence.
    cam_0 = (
        KITTI360CameraPerspective(data, seq="2013_05_28_drive_{}_sync".format(str(seq_id).zfill(4)), cam_id=0)
        if stereo_id in [-1, 0]
        else None
    )

    cam_1 = (
        KITTI360CameraPerspective(data, seq="2013_05_28_drive_{}_sync".format(str(seq_id).zfill(4)), cam_id=1)
        if stereo_id in [-1, 1]
        else None
    )

    # [3] Convert start/end image frame indexes with valid indexes
    # valid index is an image index whose camera pose data exists.
    dummy_sid = 0
    image_dir = "{}/data_2d_raw/2013_05_28_drive_{}_sync/image_{}/data_rect".format(
        data, str(seq_id).zfill(4), str(dummy_sid).zfill(2)
    )

    image_dir = os.path.join(image_dir, "*.png")
    all_image_paths = glob_data(image_dir)
    image_ids = [int(f.split("/")[-1].split(".")[0]) for f in all_image_paths]
    if cam_0 is not None:
        valid_idxs = list(cam_0.cam2world.keys())
    elif cam_1 is not None:
        if cam_0 is not None:
            assert valid_idxs == list(cam_1.cam2world.keys())
        else:
            valid_idxs = list(cam_1.cam2world.keys())
    valid_idxs = [int(i) for i in valid_idxs if int(i) < len(image_ids)]
    valid_image_ids = list(itemgetter(*valid_idxs)(image_ids))
    imgid2valididx = dict(zip(valid_image_ids, valid_idxs))

    # [3-1] Handle cases when selected image id do not contain camera poses (not valid)
    if start_img_id not in valid_image_ids:
        e_valid_image_ids = sorted(valid_image_ids + [start_img_id])
        start_valid_idx_idx = e_valid_image_ids.index(start_img_id) - 1
    else:
        start_valid_idx_idx = valid_image_ids.index(start_img_id)

    if end_img_id not in valid_image_ids:
        e_valid_image_ids = sorted(valid_image_ids + [end_img_id])
        end_valid_idx_idx = e_valid_image_ids.index(end_img_id) + 1
    else:
        end_valid_idx_idx = valid_image_ids.index(end_img_id)
    valid_idxs = valid_idxs[start_valid_idx_idx : end_valid_idx_idx + 1]

    # [4] Subsample sequence of frames using the subsampled valid indexes in [3]

    # Extract and save monocular cues of subsampled frames
    for sid, cam_obj in enumerate([cam_0, cam_1]):
        if cam_obj is not None:
            # Load & save subsampled images
            image_dir = "{}/data_2d_raw/2013_05_28_drive_{}_sync/image_{}/data_rect".format(
                data, str(seq_id).zfill(4), str(sid).zfill(2)
            )
            all_image_paths = glob_data(os.path.join(image_dir, "*.png"))
            image_filenames += list(itemgetter(*valid_idxs)(all_image_paths))

            # Load poses
            cam_poses = list(cam_obj.cam2world.values())[start_valid_idx_idx : end_valid_idx_idx + 1]
            cam2worlds += cam_poses

            # Load intrinsics
            fx += [(cam_obj.K[0, 0]) for i in range(len(valid_idxs))]
            fy += [(cam_obj.K[1, 1]) for i in range(len(valid_idxs))]
            cx += [(cam_obj.K[0, 2]) for i in range(len(valid_idxs))]
            cy += [(cam_obj.K[1, 2]) for i in range(len(valid_idxs))]

            # stereo id tracking
            stereo_ids += [sid for i in range(len(valid_idxs))]
            Ks += [cam_obj.K]

    fx = np.stack(fx)
    fy = np.stack(fy)
    cx = np.stack(cx)
    cy = np.stack(cy)
    # Note: KITTI-360 camera pose convention is T^{cam}_{world}, which projects "points" in world coordinate to camera coordinate
    # Refer to https://www.cvlibs.net/datasets/kitti-360/documentation.php
    # (The description is somewhat ambiguous, as each field of study take its own
    #  relative pose expression convention for granted)
    world2cams = [np.linalg.inv(pose) for pose in cam2worlds]
    cam2worlds = np.stack(cam2worlds)
    world2cams = np.stack(world2cams)

    # [5] Copy sub-sampled images
    saved_images_names = []
    os.makedirs(save_data / "image", exist_ok=True)
    for sid, f in zip(stereo_ids, image_filenames):
        dst_fname = save_data / "image" / ("{}_".format(sid) + (f.split("/")[-1]))
        shutil.copy(f, dst_fname)
        saved_images_names.append(str(dst_fname).split("/")[-1])

    if extract_priors:
        # [6] Extract depth
        pretrained_weights_path = omnidata_weight_dir + "omnidata_dpt_depth_v2.ckpt"  # 'omnidata_dpt_depth_v1.ckpt'
        model = DPTDepthModel(backbone="vitb_rn50_384")  # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        os.makedirs(save_data / "depth", exist_ok=True)
        for sid, f in zip(stereo_ids, image_filenames):
            depth_fname = save_data / "depth" / ("{}_".format(sid) + f.split("/")[-1])
            save_outputs(f, depth_fname, model, "depth", device)

        # [7] Extract normal
        pretrained_weights_path = omnidata_weight_dir + "omnidata_dpt_normal_v2.ckpt"
        model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        os.makedirs(save_data / "normal", exist_ok=True)
        for sid, f in zip(stereo_ids, image_filenames):
            depth_fname = save_data / "normal" / ("{}_".format(sid) + f.split("/")[-1])
            save_outputs(f, depth_fname, model, "normal", device)

    return cam2worlds, Ks, saved_images_names


def parse_camera_to_colmap(cam2worlds, Ks, image_names, colmap_dir):
    # [0] Convert cam2world to world2cam
    world2cams = np.linalg.inv(cam2worlds)
    # [1] images.txt
    os.makedirs(colmap_dir, exist_ok=True)
    img_txt = colmap_dir / "images.txt"
    with open(img_txt, "w") as f:
        for i, img in enumerate(image_names):
            t = world2cams[i, :3, 3]
            R = world2cams[i, :3, :3]
            cam_id = int(img.split("_")[0]) + 1
            qvec = rotmat2qvec(R)
            l = "{} {} {} {} {} {} {} {} {} {}\n".format(
                i + 1, qvec[0], qvec[1], qvec[2], qvec[3], t[0], t[1], t[2], cam_id, img
            )
            f.write(l)
            f.write(
                "\n"
            )  # blank line required according to (https://colmap.github.io/faq.html, reconstruct sparse model from known camera poses)

    # [2] camera.txt
    camera_txt = colmap_dir / "cameras.txt"
    w = 1408
    h = 376
    with open(camera_txt, "w") as f:
        for i, K in enumerate(Ks):
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            l = "{} PINHOLE {} {} {} {} {} {}\n".format(i + 1, w, h, fx, fy, cx, cy)
            f.write(l)

    # [3] Create empty point
    points3d_txt = colmap_dir / "points3D.txt"
    with open(points3d_txt, "w") as f:
        pass
    return None
