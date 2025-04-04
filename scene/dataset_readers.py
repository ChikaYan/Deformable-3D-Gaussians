#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON
from scipy.spatial.transform import Slerp, Rotation
from tqdm import tqdm
import skimage
import torch
from scipy import ndimage



class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    exp: Optional[np.array] = None
    depth: Optional[np.array] = None
    valid_mask: Optional[np.array] = None # contains mask [H, W] that represents valid alpha mask region of Gaussian


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    background: Optional[np.array] = None

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        # train_img = dataset_json['ids'][::4]
        # all_img = train_img
        all_id = dataset_json['ids']
        val_img = all_id[2::8]
        train_img = [i for i in all_id if i not in val_img]
        all_img = train_img + val_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]

    time_id_name = 'time_id'
    if time_id_name not in next(iter(meta_json.items()))[1]:
        time_id_name = 'warp_id'

    all_time = [meta_json[i][time_id_name] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i][time_id_name] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path)

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted(glob(os.path.join(path, 'frames/*')))
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = video_paths[i]
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def readPlenopticVideoDataset(path, eval, num_images, hold_id=[0]):
    print("Reading Training Camera")
    train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id,
                                         num_images=num_images)

    print("Reading Training Camera")
    test_cam_infos = readCamerasFromNpy(
        path, 'poses_bounds.npy', split="test", hold_id=hold_id, num_images=num_images)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3D.ply')
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfBlendShapeCameras(path, is_eval, is_debug, novel_view, only_head, is_test):
    with open(os.path.join(path, "transforms.json"), 'r') as f:
        meta_json = json.load(f)
    
    test_frames = -1_000
    frames = meta_json['frames']
    total_frames = len(frames)

    if is_test:
        frames = frames[test_frames:]
        if is_debug:
            frames = frames[-50:]
    else:
        if not is_eval:
            print(f'Loading train dataset from {path}...')
            frames = frames[0 : (total_frames + test_frames)]
            if is_debug:
                frames = frames[0: 50]
        else:
            print(f'Loading test dataset from {path}...')
            frames = frames[-50:]
            if is_debug:
                frames = frames[-50:]

    cam_infos = []
    h, w = meta_json['h'], meta_json['w']
    fx, fy, cx, cy = meta_json['fx'], meta_json['fy'], meta_json['cx'], meta_json['cy']
    fovx = focal2fov(fx, pixels=w)
    fovy = focal2fov(fy, h)

    for idx, frame in enumerate(tqdm(frames, desc="Loading data into memory in advance")):
        image_id = frame['img_id']
        image_path = os.path.join(path, "ori_imgs", str(image_id)+'.jpg')
        image = np.array(Image.open(image_path))
        if not only_head:
            mask_path = os.path.join(path, "mask", str(image_id+1)+'.png')
            seg = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)  
            # Reference MODNet colab implementation
            mask = np.repeat(np.asarray(seg)[:,:,None], 3, axis=2) / 255
        else:    
            mask_path = os.path.join(path, "parsing", str(image_id)+'.png')
            seg = cv.imread(mask_path, cv.IMREAD_UNCHANGED) 
            if seg.shape[-1] == 3:
                seg = cv.cvtColor(seg, cv.COLOR_BGR2RGB)
            else:
                seg = cv.cvtColor(seg, cv.COLOR_BGRA2RGBA)
            mask=(seg[:,:,0]==0)*(seg[:,:,1]==0)*(seg[:,:,2]==255)
            mask = np.repeat(np.asarray(mask)[:,:,None], 3, axis=2)
           
        white_background = np.ones_like(image)* 255
        image = Image.fromarray(np.uint8(image * mask + white_background * (1 - mask)))
    
        expression = np.array(frame['exp_ori']) 
        if novel_view:
            vec=np.array([0,0,0.3493212163448334])
            rot_cycle=100
            tmp_pose=np.identity(4,dtype=np.float32)
            r1 = Rotation.from_euler('y', 15+(-30)*((idx % rot_cycle)/rot_cycle), degrees=True)
            tmp_pose[:3,:3]=r1.as_matrix()
            trans=tmp_pose[:3,:3]@vec
            tmp_pose[0:3,3]=trans
            c2w = tmp_pose
        else:
            c2w = np.array(frame['transform_matrix'])
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  
        T = w2c[:3, 3]

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=fovx, FovY=fovy, image=image, image_path=image_path,
                                    image_name=image_id, width=image.size[0], height=image.size[1], exp=expression, 
                                    fid=image_id
                                    ))
    '''finish load all data'''
    return cam_infos

def readNeRFBlendShapeDataset(path, eval, is_debug, novel_view, only_head, is_test):
    print("Load NeRFBlendShape Train Dataset")
    train_cam_infos = readNerfBlendShapeCameras(path=path, is_eval=False, is_debug=is_debug, novel_view=novel_view, only_head=only_head, is_test=False)
    print("Load NeRFBlendShape Test Dataset")
    test_cam_infos = readNerfBlendShapeCameras(path=path, is_eval=eval, is_debug=is_debug, novel_view=novel_view, only_head=only_head, is_test=is_test)

    if not eval: 
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    '''Init point cloud'''
    if not os.path.exists(ply_path):
        # Since mono dataset has no colmap, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd, 
                           train_cameras=train_cam_infos, 
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, 
                           ply_path=ply_path)

    return scene_info


def readNerSembleCameras(path, is_eval, is_debug, novel_view):
    with open(os.path.join(path, "transforms.json"), 'r') as f:
        meta_json = json.load(f)
    
    test_frames = -1_000
    frames = meta_json['frames']
    total_frames = len(frames)
    
    if not is_eval:
        print(f'Loading train dataset from {path}...')
        frames = frames[0 : (total_frames + test_frames)]
        if is_debug:
            frames = frames[0: 50]
    else:
        print(f'Loading test dataset from {path}...')
        frames = frames[-50:]
        if is_debug:
            frames = frames[-50:]

    cam_infos = []
    h, w = meta_json['h'], meta_json['w']
    fx, fy, cx, cy = meta_json['fx'], meta_json['fy'], meta_json['cx'], meta_json['cy']
    fovx = focal2fov(fx, pixels=w)
    fovy = focal2fov(fy, h)

    for idx, frame in enumerate(tqdm(frames, desc="Loading data into memory in advance")):
        image_id = frame['img_id']
        image_path = os.path.join(path, "ori_imgs", str(image_id)+'.jpg')
        image = np.array(Image.open(image_path))
        if not only_head:
            mask_path = os.path.join(path, "mask", str(image_id+1)+'.png')
            seg = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)  
            # Reference MODNet colab implementation
            mask = np.repeat(np.asarray(seg)[:,:,None], 3, axis=2) / 255
        else:    
            mask_path = os.path.join(path, "parsing", str(image_id)+'.png')
            seg = cv.imread(mask_path, cv.IMREAD_UNCHANGED) 
            if seg.shape[-1] == 3:
                seg = cv.cvtColor(seg, cv.COLOR_BGR2RGB)
            else:
                seg = cv.cvtColor(seg, cv.COLOR_BGRA2RGBA)
            mask=(seg[:,:,0]==0)*(seg[:,:,1]==0)*(seg[:,:,2]==255)
            mask = np.repeat(np.asarray(mask)[:,:,None], 3, axis=2)
           
        white_background = np.ones_like(image)* 255
        image = Image.fromarray(np.uint8(image * mask + white_background * (1 - mask)))
    
        expression = np.array(frame['exp_ori']) 
        if novel_view:
            vec=np.array([0,0,0.3493212163448334])
            rot_cycle=100
            tmp_pose=np.identity(4,dtype=np.float32)
            r1 = Rotation.from_euler('y', 15+(-30)*((idx % rot_cycle)/rot_cycle), degrees=True)
            tmp_pose[:3,:3]=r1.as_matrix()
            trans=tmp_pose[:3,:3]@vec
            tmp_pose[0:3,3]=trans
            c2w = tmp_pose
        else:
            c2w = np.array(frame['transform_matrix'])
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  
        T = w2c[:3, 3]

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=fovx, FovY=fovy, image=image, image_path=image_path,
                                    image_name=image_id, width=image.size[0], height=image.size[1], exp=expression, 
                                    fid=image_id
                                    ))
    '''finish load all data'''
    return cam_infos


def readNerSembleDataset(path, eval, is_debug, novel_view):
    print("Load NeRFBlendShape Train Dataset")
    train_cam_infos = readNerSembleCameras(path=path, is_eval=False, is_debug=is_debug, novel_view=novel_view)
    print("Load NeRFBlendShape Test Dataset")
    test_cam_infos = readNerSembleCameras(path=path, is_eval=eval, is_debug=is_debug, novel_view=novel_view)

    if not eval: 
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    '''Init point cloud'''
    if not os.path.exists(ply_path):
        # Since mono dataset has no colmap, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd, 
                           train_cameras=train_cam_infos, 
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, 
                           ply_path=ply_path)

    return scene_info


def readNerfaceCameras(path, is_eval, is_debug, novel_view, is_test=False):
    
    if is_eval:
        fname = "transforms_val.json"
    else:
        fname = "transforms_train.json"

    if is_test:
        fname = "transforms_test.json"

    with open(os.path.join(path, fname), 'r') as f:
        meta_json = json.load(f)
    
    frames = meta_json['frames']
    total_frames = len(frames)
    
    if not is_eval:
        if is_debug:
            frames = frames[0: 50]
        print(f'Loading train dataset from {path}, len {len(frames)}...')
    else:
        frames = frames[-50:]
        print(f'Loading test dataset from {path}, len {len(frames)}...')

    cam_infos = []
    h, w = 512, 512
    camera_angle_x = float(meta_json["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    fx, fy, cx, cy = focal, focal, 0.5, 0.5
    fovx = focal2fov(fx, pixels=w)
    fovy = focal2fov(fy, h)

    for idx, frame in enumerate(tqdm(frames, desc="Loading data into memory in advance")):
        image_id_str = frame['file_path'].strip("./")
        image_path = os.path.join(path, str(image_id_str)+'.png')
        image = Image.fromarray(np.array(Image.open(image_path)))

        image_id = int(image_id_str[-4:])
        if is_eval or is_test:
            image_id = 0
        # if not only_head:
        #     mask_path = os.path.join(path, "mask", str(image_id+1)+'.png')
        #     seg = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)  
        #     # Reference MODNet colab implementation
        #     mask = np.repeat(np.asarray(seg)[:,:,None], 3, axis=2) / 255
        # else:    
        #     mask_path = os.path.join(path, "parsing", str(image_id)+'.png')
        #     seg = cv.imread(mask_path, cv.IMREAD_UNCHANGED) 
        #     if seg.shape[-1] == 3:
        #         seg = cv.cvtColor(seg, cv.COLOR_BGR2RGB)
        #     else:
        #         seg = cv.cvtColor(seg, cv.COLOR_BGRA2RGBA)
        #     mask=(seg[:,:,0]==0)*(seg[:,:,1]==0)*(seg[:,:,2]==255)
        #     mask = np.repeat(np.asarray(mask)[:,:,None], 3, axis=2)
           
        # white_background = np.ones_like(image)* 255
        # image = Image.fromarray(np.uint8(image * mask + white_background * (1 - mask)))
    
        expression = np.array(frame['expression']) 
        if novel_view:
            print('novel view function for nerface not tested!')
            vec=np.array([0,0,0.3493212163448334])
            rot_cycle=100
            tmp_pose=np.identity(4,dtype=np.float32)
            r1 = Rotation.from_euler('y', 15+(-30)*((idx % rot_cycle)/rot_cycle), degrees=True)
            tmp_pose[:3,:3]=r1.as_matrix()
            trans=tmp_pose[:3,:3]@vec
            tmp_pose[0:3,3]=trans
            c2w = tmp_pose
        else:
            c2w = np.array(frame['transform_matrix'])
        # c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  
        T = w2c[:3, 3]

        # matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
        # R = -np.transpose(matrix[:3, :3])
        # R[:, 0] = -R[:, 0]
        # T = -matrix[:3, 3]

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=fovx, FovY=fovy, image=image, image_path=image_path,
                                    image_name=image_id_str, width=image.size[0], height=image.size[1], exp=expression, 
                                    fid=image_id
                                    ))
    '''finish load all data'''
    return cam_infos

def readNerfaceDataset(path, eval, is_debug, novel_view, is_test):
    print("Load Nerface Train Dataset")
    train_cam_infos = readNerfaceCameras(path=path, is_eval=False, is_debug=is_debug, novel_view=novel_view, is_test=False)
    print("Load Nerface Test Dataset")
    test_cam_infos = readNerfaceCameras(path=path, is_eval=eval, is_debug=is_debug, novel_view=novel_view, is_test=is_test)

    if not eval: 
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    '''Init point cloud'''
    if not os.path.exists(ply_path):
        # Since mono dataset has no colmap, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # load background
    background = np.array(Image.open(os.path.join(path, 'bg', '00001.png')))

    scene_info = SceneInfo(point_cloud=pcd, 
                           train_cameras=train_cam_infos, 
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, 
                           ply_path=ply_path,
                           background=background,
                           )

    return scene_info


def readIMAvatarCameras_(path, is_eval, is_debug, novel_view, is_test=False):
    
    if is_eval or is_test:
        sub_dir = ['MVI_1812']
        subsample = 200
    else:
        sub_dir = ['MVI_1810', 'MVI_1814']
        subsample = 1

    img_res = [512,512]
    h, w = img_res
    cam_infos = []

    image_id = 0
    for dir in sub_dir:
        instance_dir = os.path.join(path, dir)
        assert os.path.exists(instance_dir), "Data directory is empty"

        cam_file = '{0}/{1}'.format(instance_dir, 'flame_params.json')

        with open(cam_file, 'r') as f:
            camera_dict = json.load(f)
        
        focal_cxcy = camera_dict['intrinsics']
        # construct intrinsic matrix in pixels
        intrinsics = np.eye(3)
        if focal_cxcy[3] > 1:
            # An old format of my datasets...
            intrinsics[0, 0] = focal_cxcy[0] * img_res[0] / 512
            intrinsics[1, 1] = focal_cxcy[1] * img_res[1] / 512
            intrinsics[0, 2] = focal_cxcy[2] * img_res[0] / 512
            intrinsics[1, 2] = focal_cxcy[3] * img_res[1] / 512
        else:
            intrinsics[0, 0] = focal_cxcy[0] * img_res[0]
            intrinsics[1, 1] = focal_cxcy[1] * img_res[1]
            intrinsics[0, 2] = focal_cxcy[2] * img_res[0]
            intrinsics[1, 2] = focal_cxcy[3] * img_res[1]

        fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
        fovx = focal2fov(fx, pixels=w)
        fovy = focal2fov(fy, h)

        for idx in range(0, len(camera_dict['frames']), subsample):
            frame = camera_dict['frames'][idx]
            # world to camera matrix
            world_mat = np.array(frame['world_mat']).astype(np.float32)
            # camera to world matrix
            pose = load_K_Rt_from_P(None, world_mat)[1]
            a = pose[0:1, :]
            b = pose[1:2, :]
            c = pose[2:3, :]
            pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)
            S = np.eye(3)
            S[1, 1] = -1
            S[2, 2] = -1
            pose[1, 3] = -pose[1, 3]
            pose[2, 3] = -pose[2, 3]
            pose[:3, :3] = S @ pose[:3, :3] @ S
            a = pose[0:1, :]
            b = pose[1:2, :]
            c = pose[2:3, :]
            pose = np.concatenate([a, c, b, pose[3:, :]], 0)
            pose[:, 3] *= 0.5
            matrix = np.linalg.inv(pose)
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]


            expression = np.array(frame['expression']).astype(np.float32)
            flame_pose = np.array(frame['pose']).astype(np.float32) # not sure if needed
            image_path = '{0}/{1}.png'.format(instance_dir, frame["file_path"])
            mask_path = image_path.replace('image', 'mask')
            # if use_semantics:
            #     semantic_paths = image_path.replace('image', 'semantic')
            img_name = int(frame["file_path"].split('/')[-1])
            # bbox = ((np.array(frame['bbox']) + 1.) * np.array([img_res[0],img_res[1],img_res[0],img_res[1]])/ 2).astype(int)

            mask = imageio.imread(mask_path, as_gray=True)
            mask = skimage.img_as_float32(mask)
            mask = mask > 127.5

            image = np.array(Image.open(image_path))
            image[~mask] = 255
            image = Image.fromarray(image)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=fovx, FovY=fovy, image=image, image_path=image_path,
                                image_name=img_name, width=image.size[0], height=image.size[1], exp=expression, 
                                fid=image_id
                                ))
            if not is_eval and not is_test:
                image_id += 1

            if is_debug and image_id > 50:
                break

    # shape_params = np.array(camera_dict['shape_params']).astype(np.float32).unsqueeze(0)
            
def readIMAvatarCameras(path, is_eval, is_debug, novel_view, is_test=False):
    
    if is_eval or is_test:
        sub_dir = ['test']
        # sub_dir = ['MVI_1812']
        subsample = 200
        if is_test:
            subsample = 1
    else:
        sub_dir = ['train']
        # sub_dir = ['MVI_1810', 'MVI_1814']
        subsample = 1

    img_res = [512,512]
    h, w = img_res
    cam_infos = []

    image_id = 0
    INTRI_REVERT_FLAG = False
    for dir in sub_dir:
        instance_dir = os.path.join(path, dir)
        assert os.path.exists(instance_dir), "Data directory is empty"

        cam_file = '{0}/{1}'.format(instance_dir, 'flame_params.json')

        with open(cam_file, 'r') as f:
            camera_dict = json.load(f)
        
        focal_cxcy = camera_dict['intrinsics']
        # construct intrinsic matrix in pixels
        intrinsics = np.zeros((4, 4))

        # # from whatever camera convention to pytorch3d
        # intrinsics[0, 0] = focal_cxcy[0] * 2
        # intrinsics[1, 1] = focal_cxcy[1] * 2
        # intrinsics[0, 2] = (focal_cxcy[2] * 2 - 1.0) * -1
        # intrinsics[1, 2] = (focal_cxcy[3] * 2 - 1.0) * -1

        if focal_cxcy[3] > 1:
            # An old format of my datasets...
            intrinsics[0, 0] = focal_cxcy[0] * img_res[0] / 512
            intrinsics[1, 1] = focal_cxcy[1] * img_res[1] / 512
            intrinsics[0, 2] = focal_cxcy[2] * img_res[0] / 512
            intrinsics[1, 2] = focal_cxcy[3] * img_res[1] / 512
        else:
            intrinsics[0, 0] = focal_cxcy[0] * img_res[0]
            intrinsics[1, 1] = focal_cxcy[1] * img_res[1]
            intrinsics[0, 2] = focal_cxcy[2] * img_res[0]
            intrinsics[1, 2] = focal_cxcy[3] * img_res[1]

        if intrinsics[0, 0] < 0:
            intrinsics[:, 0] *= -1
            INTRI_REVERT_FLAG = True

        fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
        fovx = focal2fov(fx, pixels=w)
        fovy = focal2fov(fy, h)

        for idx in range(0, len(camera_dict['frames']), subsample):
            frame = camera_dict['frames'][idx]
            # world to camera matrix
            world_mat = np.array(frame['world_mat']).astype(np.float32)
            # camera to world matrix
            if INTRI_REVERT_FLAG:
                world_mat[0, :] *= -1
            world_mat[:3, 2] *= -1
            world_mat[2, 3] *= -1
            # # world_mat is now in Pytorch3D format
            # c2w = np.linalg.inv(np.concatenate([world_mat, np.array([[0,0,0,1]])],axis=0))
            # # convert from Pytorch3D (X left Y up Z forward) to OpenCV/Colmap (X right Y down Z forward)
            # c2w[:3, 0:2] *= -1

            # # get the world-to-camera transform and set R, T
            # w2c = np.linalg.inv(c2w)
            # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code

            T = world_mat[:3, 3]
            T[:2] *= -1.

            expression = np.array(frame['expression']).astype(np.float32)
            flame_pose = np.array(frame['pose']).astype(np.float32)
            
            # convert flame global & head rotation to c2w
            rot_mats = batch_rodrigues(torch.from_numpy(flame_pose[:6]).view([-1, 3])).numpy()
            c2w_rot = rot_mats[0] @ rot_mats[1]
            w2c_rot = np.linalg.inv(c2w_rot)
            R = np.transpose(w2c_rot) 


            image_path = '{0}/{1}.png'.format(instance_dir, frame["file_path"])
            mask_path = image_path.replace('image', 'mask')
            # if use_semantics:
            #     semantic_paths = image_path.replace('image', 'semantic')
            img_name = int(frame["file_path"].split('/')[-1])
            # bbox = ((np.array(frame['bbox']) + 1.) * np.array([img_res[0],img_res[1],img_res[0],img_res[1]])/ 2).astype(int)

            mask = imageio.imread(mask_path, as_gray=True)
            mask = skimage.img_as_float32(mask)
            mask = mask > 127.5

            image = np.array(Image.open(image_path))
            image[~mask] = 255
            image = Image.fromarray(image)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=fovx, FovY=fovy, image=image, image_path=image_path,
                                image_name=img_name, width=image.size[0], height=image.size[1], exp=np.concatenate([expression, flame_pose]), 
                                fid=image_id
                                ))
            if not is_eval and not is_test:
                image_id += 1

            if is_debug and image_id > 50:
                break

    # shape_params = np.array(camera_dict['shape_params']).astype(np.float32).unsqueeze(0)


    return cam_infos

def readIMAvatarDataset(path, eval, is_debug, novel_view, is_test):
    print("Load IMAvatar Train Dataset")
    train_cam_infos = readIMAvatarCameras(path=path, is_eval=False, is_debug=is_debug, novel_view=novel_view, is_test=False)
    print("Load IMAvatar Test Dataset")
    test_cam_infos = readIMAvatarCameras(path=path, is_eval=eval, is_debug=is_debug, novel_view=novel_view, is_test=is_test)

    if not eval: 
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    '''Init point cloud'''
    if not os.path.exists(ply_path):
        # Since mono dataset has no colmap, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # load background
    # background = np.array(Image.open(os.path.join(path, 'bg', '00001.png')))

    scene_info = SceneInfo(point_cloud=pcd, 
                           train_cameras=train_cam_infos, 
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, 
                           ply_path=ply_path,
                           background=None,
                           )

    return scene_info

def readInstaCameras(path, is_eval, is_debug, novel_view, is_test=False):
    with open(os.path.join(path, "transforms.json"), 'r') as f:
        meta_json = json.load(f)
    
    with open(os.path.join(path, "split.json"), 'r') as f:
        split_json = json.load(f)

    if is_test:
        frame_names = split_json['test']
    elif is_eval:
        frame_names = split_json['test'][:50]
    else:
        frame_names = split_json['train']
        
    if is_debug:
        frame_names = frame_names[:50]

    
    frames = meta_json['frames']
    frames = sorted(frames, key=lambda x: int(Path(x['file_path']).stem))
    total_frames = len(frame_names)



    cam_infos = []
    h, w = meta_json['h'], meta_json['w']
    fx, fy, cx, cy = meta_json['fl_x'], meta_json['fl_y'], meta_json['cx'], meta_json['cy']
    fovx = focal2fov(fx, w)
    fovy = focal2fov(fy, h)

    for idx, frame in enumerate(tqdm(frames, desc="Loading data into memory in advance")):
        if Path(frame['file_path']).name not in frame_names:
            continue
        
        USE_MATTED_GT = False
        # USE_MATTED_GT = True

        if USE_MATTED_GT:
            image_path = os.path.join(path, frame['file_path'].replace('images', 'matted'))
            image_id = int(Path(image_path).stem)
            image = np.array(Image.open(image_path))

            alpha_mask = image[..., 3:] / 255.
            image = image[..., :3]   
            white_background = np.ones_like(image)* 255
            image = Image.fromarray(np.uint8(image * alpha_mask + white_background * (1 - alpha_mask)))
        else:
            image_path = os.path.join(path, frame['file_path'].replace('images', 'background'))
            image_id = int(Path(image_path).stem)
            image = Image.open(image_path)

            # we still need alpha mask for valid mask and alpha loss
            alpha_mask = np.array(Image.open(image_path.replace('background', 'matted')))[..., 3:] / 255.


        # read parsing mask
        parsing_mask =  np.array(Image.open(os.path.join(path, frame['file_path'].replace('images', 'seg_mask'))))[..., :1] 
        parsing_mask = (parsing_mask == 90)
        parsing_mask = (parsing_mask > 0).astype(np.int32)
        parsing_mask = ndimage.median_filter(parsing_mask, size=5)
        parsing_mask = (ndimage.binary_dilation(parsing_mask, iterations=3) > 0).astype(np.uint8)

        head_mask = ndimage.median_filter(alpha_mask, size=5)
        head_mask = np.where(parsing_mask, np.zeros_like(head_mask), head_mask)
        head_mask[head_mask > 0.5] = 1.



        exp_path = os.path.join(path, frame['exp_path'])
        expression = np.array(np.loadtxt(exp_path)) # TODO: INSTA uses [16:] of this expression. Not sure why
        eyes = np.array(np.loadtxt(exp_path.replace('exp', 'eyes')))
        jaw = np.array(np.loadtxt(exp_path.replace('exp', 'jaw')))

        expression = np.concatenate([expression,eyes,jaw])

        if novel_view:
            # vec=np.array([0,0,0.3493212163448334])
            # rot_cycle=100
            # tmp_pose=np.identity(4,dtype=np.float32)
            # r1 = Rotation.from_euler('y', 15+(-30)*((idx % rot_cycle)/rot_cycle), degrees=True)
            # tmp_pose[:3,:3]=r1.as_matrix()
            # trans=tmp_pose[:3,:3]@vec
            # tmp_pose[0:3,3]=trans
            # c2w = tmp_pose
            pass
        else:
            c2w = np.array(frame['transform_matrix'])
        # c2w[:3, 1:3] *= -1
        # pose from INSTA pipeline seems to have already converted to INST-NGP (OpenCV/COLMAP) format
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  
        T = w2c[:3, 3]

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=fovx, FovY=fovy, image=image, image_path=image_path,
                                    image_name=image_id, width=image.size[0], height=image.size[1], exp=expression, 
                                    fid=image_id, valid_mask=head_mask[...,0]
                                    ))
    '''finish load all data'''
    return cam_infos


    return cam_infos

def readInstaDataset(path, eval, is_debug, novel_view, is_test):
    print("Load Insta Train Dataset")
    train_cam_infos = readInstaCameras(path=path, is_eval=False, is_debug=is_debug, novel_view=novel_view, is_test=False)
    print("Load Insta Test Dataset")
    test_cam_infos = readInstaCameras(path=path, is_eval=eval, is_debug=is_debug, novel_view=novel_view, is_test=is_test)

    if not eval: 
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    '''Init point cloud'''
    if not os.path.exists(ply_path):
        # Since mono dataset has no colmap, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # load background
    # background = np.array(Image.open(os.path.join(path, 'bg', '00001.png')))

    scene_info = SceneInfo(point_cloud=pcd, 
                           train_cameras=train_cam_infos, 
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, 
                           ply_path=ply_path,
                           background=None,
                           )

    return scene_info



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "nerfblendshape": readNeRFBlendShapeDataset,  
    "nersemble": readNerSembleDataset,  
    "nerface": readNerfaceDataset,  
    "imavatar": readIMAvatarDataset,  
    "insta": readInstaDataset,  
}
