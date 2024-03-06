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

import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.refine_model import RefineModel
import imageio
import numpy as np
import pickle
from argparse import Namespace
from scene.layer_model import LayerModel


try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False
WANDB_FOUND = False


def render_set(
        model_path, 
        load2gpu_on_the_fly, 
        is_6dof, 
        name, 
        iteration, 
        views, 
        gaussians, 
        scene,
        pipeline, 
        background, 
        deform, 
        deform_sh, 
        refine_model=None, 
        layer_model=None,
        model_args=None,
        args=None,
        ):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    pre_refine_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pre_refine")
    refine_path = os.path.join(model_path, name, "ours_{}".format(iteration), "refine")

    bg_path = os.path.join(model_path, name, "ours_{}".format(iteration), "layer_bg")
    fg_path = os.path.join(model_path, name, "ours_{}".format(iteration), "layer_fg")


    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(pre_refine_path, exist_ok=True)
    if refine_model is not None:
        makedirs(refine_path, exist_ok=True)
    if layer_model is not None:
        makedirs(bg_path, exist_ok=True)
        makedirs(fg_path, exist_ok=True)


    imgs = []
    bgs = []

    for idx, view_cam in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view_cam.load2device()
        fid = view_cam.fid
        exp = view_cam.exp
        # if idx == 0:
        #     fid = view_cam.fid
        #     exp = view_cam.exp
        # else:
        #     pass
        xyz = gaussians.get_xyz
        time_input = exp.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling, d_sh = deform.step(xyz.detach(), time_input)
        if not deform_sh:
            d_sh = None
        results = render(view_cam, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, d_sh, render_ex_feature=model_args.use_ex_feature)
        rendering = results["render"]
        depth = results["depth"]
        alpha = results["alpha"]
        depth = depth / (depth.max() + 1e-5)

        if not args.no_extra:
            torchvision.utils.save_image(rendering, os.path.join(pre_refine_path, '{0:05d}'.format(idx) + ".png"))

        if refine_model is not None:
            feature_im = results["feature_im"] # [h, w, EX_FEATURE_DIM]
            exp_input = exp.unsqueeze(0)
            time_input = fid.unsqueeze(0)

            refined_rgb = refine_model.step(feature_im, exp_input, time=time_input)[0]

            if not args.no_extra:
                torchvision.utils.save_image(refined_rgb, os.path.join(refine_path, '{0:05d}'.format(idx) + ".png"))

            if model_args.refine_mode == 'add':
                rendering = torch.clamp(rendering + refined_rgb, 0, 1)
            elif model_args.refine_mode == 'replace':
                if model_args.refine_parser_type == 'pix2pix':
                    refined_rgb = (refined_rgb + 1.) / 2.
                rendering = torch.clamp(refined_rgb, 0, 1)
            else:
                raise NotImplementedError(f"refine mode {model_args.refine_mode} not supported")
            

        # apply background
        fg = bg = None
        if layer_model is not None:
            time_input = fid.unsqueeze(0)
            exp_input = exp.unsqueeze(0)
            cam_pose_input = torch.from_numpy(np.concatenate([view_cam.T, view_cam.look_at[:,0]])).type_as(exp_input).unsqueeze(0)

            bg = layer_model.get_background(time_input, exp_input, cam_pose_input)
            if layer_model.fg_model is not None:
                # blend foreground & background
                fg = layer_model.get_background(time_input, exp_input, cam_pose_input) # [4, H, W]
                rendering = fg[3:] * fg[:3] + (1.-fg[3:]) * alpha * rendering + (1.-fg[3:]) * (1.-alpha) * bg
            else:
                rendering = rendering + (1.-alpha) * bg
                # rendering = bg
            bgs.append(bg)
                
        elif scene.background is not None:
            # composite with given background image
            # only do so when layer model is not used
            scene_background = torch.from_numpy(scene.background).permute([2,0,1]) / 255.
            scene_background = scene_background.to('cuda')
            rendering = rendering + (1.-alpha) * scene_background
        else:
            # use black or white bacground
            if model_args.white_background:
                rendering = rendering + (1.-alpha) * 1.
        rendering = torch.clamp(rendering, 0, 1)

        gt = view_cam.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if not args.no_extra:
            # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
            if fg is not None:
                torchvision.utils.save_image(fg, os.path.join(fg_path, '{0:05d}'.format(idx) + ".png"))
            if bg is not None:
                torchvision.utils.save_image(bg, os.path.join(bg_path, '{0:05d}'.format(idx) + ".png"))

        imgs.append(rendering)
    
    if not args.no_vid:
        imgs = torch.stack(imgs).permute([0,2,3,1]).cpu().numpy()
        imgs = (imgs * 255).astype(np.uint8)
        torchvision.io.write_video(os.path.join(model_path, name, "ours_{}".format(iteration), "renders.mp4"), imgs, fps=30)

        if len(bgs) > 0:
            bgs = torch.stack(bgs).permute([0,2,3,1]).cpu().numpy()
            bgs = (bgs * 255).astype(np.uint8)
            torchvision.io.write_video(os.path.join(model_path, name, "ours_{}".format(iteration), "bg.mp4"), bgs, fps=30)


def interpolate_time(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, deform_sh):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling, d_sh = deform.step(xyz.detach(), time_input)
        if not deform_sh:
            d_sh = None
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, d_sh)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_view(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer, deform_sh):
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    render_poses = torch.stack(render_wander_path(view), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling, d_sh = timer.step(xyz.detach(), time_input)
        if not deform_sh:
            d_sh = None
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, d_sh)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        # acc = results["acc"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(acc, os.path.join(acc_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, deform_sh):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling, d_sh = deform.step(xyz.detach(), time_input)
        if not deform_sh:
            d_sh = None
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof, d_sh)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_poses(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer):
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)
    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def interpolate_view_original(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background,
                              timer):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def render_sets(model_args: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str, args):
    with torch.no_grad():
        gaussians = GaussianModel(model_args.sh_degree, model_args.use_ex_feature)
        model_args.is_test = True
        scene = Scene(model_args, gaussians, load_iteration=iteration, shuffle=False)
        exp_dim = scene.train_cameras[1.0][0].exp.shape[-1]
        deform = DeformModel(t_dim=exp_dim, is_blender=model_args.is_blender, is_6dof=model_args.is_6dof)
        deform.load_weights(model_args.model_path)

        if model_args.use_ex_feature:
            refine_model = RefineModel(
                exp_dim=exp_dim,
                feature_dim=gaussians.EX_FEATURE_DIM, 
                exp_multires=0, 
                out_rescale=model_args.refine_out_rescale,
                parser_type=model_args.refine_parser_type,
                pix2pix_n_blocks=model_args.pix2pix_n_blocks,
                stylegan_n_blocks=model_args.stylegan_n_blocks,
                stylegan_noise=model_args.stylegan_noise,
                )
            refine_model.load_weights(model_args.model_path)
            refine_model.stylegan_noise = 'none' # disable noise after weight loading to avoid error
        else:
            refine_model = None

        if model_args.layer_model != 'none':
            layer_model = LayerModel(
                model_args.layer_model,
                img_size=(scene.train_cameras[1.0][0].image_height, scene.train_cameras[1.0][0].image_width),
                exp_dim=exp_dim,
                feature_dim=model_args.layer_feature_dim, 
                layer_parser_rescale=model_args.layer_out_rescale,
                parser_type=model_args.layer_parser_type,
                exp_multires=model_args.layer_input_exp_multires,
                t_multires=model_args.layer_input_t_multires,
                pose_multires=model_args.layer_input_pose_multires,
                layer_encoding=model_args.layer_encoding,
                )
            layer_model.load_weights(model_args.model_path)
        else:
            layer_model = None

        # bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(model_args.model_path, model_args.load2gpu_on_the_fly, model_args.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras()[::50], gaussians, scene, pipeline,
                        background, deform, model_args.deform_sh, refine_model=refine_model, layer_model=layer_model, model_args=model_args, args=args)

        if not skip_test:
            render_func(model_args.model_path, model_args.load2gpu_on_the_fly, model_args.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, scene, pipeline,
                        background, deform, model_args.deform_sh, refine_model=refine_model, layer_model=layer_model, model_args=model_args, args=args)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_vid", action="store_true")
    parser.add_argument("--no_extra", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    with open(args.model_path  + '/cfg_args', 'r') as f:
        ns = f.read()
        print(ns)
        saved_args = parser.parse_args(namespace=eval(ns))


        for k in saved_args.__dict__:
            if saved_args.__dict__[k] is None and hasattr(model, k):
                saved_args.__dict__[k] = getattr(model, k)
        # import pdb; pdb.set_trace()
        saved_args.__dict__['is_debug'] = args.is_debug

    render_sets(model.extract(saved_args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, args)
