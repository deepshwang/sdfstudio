# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for friends dataset"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from glob import glob
from operator import itemgetter
from pathlib import Path
from typing import Dict, Literal, Optional, Type

import cv2
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torchtyping import TensorType

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.kitti360camera import KITTI360CameraPerspective
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox

# random rotate
# from scipy.spatial.transform import Rotation
# random_rotation = torch.eye(4)
# random_rotation[:3, :3] = torch.from_numpy(Rotation.random().as_matrix())


CONSOLE = Console()


def get_src_from_pairs(
    ref_idx, all_imgs, pairs_srcs, neighbors_num=None, neighbors_shuffle=False
) -> Dict[str, TensorType]:
    # src_idx[0] is ref img
    src_idx = pairs_srcs[ref_idx]
    # randomly sample neighbors
    if neighbors_num and neighbors_num > -1 and neighbors_num < len(src_idx) - 1:
        if neighbors_shuffle:
            perm_idx = torch.randperm(len(src_idx) - 1) + 1
            src_idx = torch.cat([src_idx[[0]], src_idx[perm_idx[:neighbors_num]]])
        else:
            src_idx = src_idx[: neighbors_num + 1]
    src_idx = src_idx.to(all_imgs.device)
    return {"src_imgs": all_imgs[src_idx], "src_idxs": src_idx}


def get_image(image_filename, alpha_color=None) -> TensorType["image_height", "image_width", "num_channels"]:
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def get_depths_and_normals(image_idx: int, depths=None, normals=None):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    out_dict = {}
    # depth
    if depths is not None:
        depth = depths[image_idx]
        out_dict["depth"] = depth
    # normal
    if normals is not None:
        normal = normals[image_idx]
        out_dict["normal"] = normal

    return out_dict


@dataclass
class KITTI360DataParserConfig(DataParserConfig):
    """
    Scene dataset parser configurations
    """

    _target: Type = field(default_factory=lambda: KITTI360)
    """target class to instantiate"""
    data: Path = Path("data/KITTI360")
    """Directory specifying location of processed data."""
    include_depth_prior: bool = True
    """whether or not to include depth prior """
    include_normal_prior: bool = True
    """whether or not to include normal prior """
    include_flow_prior: bool = False
    """Whether or not to include optical flow prior"""
    downscale_factor: int = 1
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Friends axis-aligned bbox will be scaled to this value.
    """
    seq_id: int = None
    """KITTI-360 sequence ID (required)"""
    stereo_id: int = -1
    """Index of stereo cameras to use. -1 indicates to use both cameras. Use both cameras by default"""
    start_img_id: int = None
    end_img_id: int = None
    """
    Subset of images expressed with range of image ids. 
    The ids can be selected from image filename, 
    e.g., 0000000176.png -> 176.
    These arguments are required since it's too much to load full data of a sequence. 
    """


@dataclass
class KITTI360(DataParser):
    """KITTI360 Dataset"""

    config: KITTI360DataParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths

        # [1] Set pre-processed dataset path & assert to exist.
        data_path = (
            self.config.data
            / f"seq_{str(self.config.seq_id).zfill(4)}"
            / (
                f"stereo_{self.config.stereo_id}_"
                + f"{str(self.config.start_img_id).zfill(10)}-"
                + str(self.config.end_img_id).zfill(10)
            )
        )
        print(data_path)
        assert os.path.exists(data_path), f"Preprocessed data not stored in: {self.config.data}. Please check"

        # [2] Load paths for GTs / priors
        image_paths = glob_data(str(data_path / "image" / "*.png"))
        depth_paths = glob_data(str(data_path / "depth" / "*.npy"))
        normal_paths = glob_data(str(data_path / "normal" / "*.npy"))
        n_images = len(image_paths)

        # [3] Load camera
        cam_file = data_path / "cameras.npz"
        camera_dict = np.load(cam_file)
        pose_all = [torch.from_numpy(camera_dict["cam2scaled_%d" % (i)]).float() for i in range(n_images)]
        intrinsics_all = [torch.from_numpy(camera_dict["intrinsics_%d" % (i)]).float() for i in range(n_images)]

        image_filenames = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for idx in range(n_images):
            # unpack data
            image_filename = image_paths[idx]
            # TODO now we has the first intrincis
            intrinsics = intrinsics_all[idx]
            camtoworld = pose_all[idx]
            # append data
            image_filenames.append(image_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)
        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        if self.config.include_depth_prior:
            # load monocular depths and normals
            depth_images = []
            for idx, dpath in enumerate(depth_paths):
                depth = np.load(dpath)
                depth_images.append(torch.from_numpy(depth).float())

            depth_images = torch.stack(depth_images)

        else:
            depth_images = None

        if self.config.include_normal_prior:
            normal_images = []
            for idx, npath in enumerate(normal_paths):
                normal = np.load(npath)

                # important as the output of omnidata is normalized
                normal = normal * 2.0 - 1.0
                normal = torch.from_numpy(normal).float()

                # transform normal to world coordinate system
                rot = camera_to_worlds[idx][:3, :3].clone()
                normal_map = normal.reshape(3, -1)
                normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

                normal_map = rot @ normal_map
                normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
                normal_images.append(normal_map)

            normal_images = torch.stack(normal_images)

        else:
            normal_images = None

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        scene_box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32))

        height, width = get_image(image_filenames[0]).shape[:2]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        additional_inputs_dict = {}
        if self.config.include_depth_prior or self.config.include_normal_prior:
            additional_inputs_dict["cues"] = {
                "func": get_depths_and_normals,
                "kwargs": {"depths": depth_images, "normals": normal_images},
            }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs=additional_inputs_dict,
            depths=depth_images,
            normals=normal_images,
        )
        return dataparser_outputs
