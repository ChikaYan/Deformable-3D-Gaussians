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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from scene.cameras import Camera
from scene.refine_model import RefineModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from pathlib import Path
from scene.layer_model import LayerModel
from utils.general_utils import get_num_trainable_params
from scene.cameras import Camera
from typing import Optional
import numpy as np

# try:
#     from torch.utils.tensorboard import SummaryWriter

#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False

try:
    import wandb

    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

def training(model_args:ModelParams, opt:OptimizationParams, pipe:PipelineParams, testing_iterations, saving_iterations):
    prepare_output_and_logger(model_args, opt, pipe)
    gaussians = GaussianModel(model_args.sh_degree, model_args.use_ex_feature)

    scene = Scene(model_args, gaussians)

    t_dim = scene.train_cameras[1.0][0].exp.shape[-1]

    model_summary = "######### Num of Trainable Parameters #########\n"
    deform = DeformModel(t_dim=t_dim, is_blender=model_args.is_blender, is_6dof=model_args.is_6dof)
    deform.train_setting(opt)

    deform_n_param = get_num_trainable_params(deform.deform)
    model_summary += f"{'Deform'.ljust(20)}: {deform_n_param}\n"
    if WANDB_FOUND:
        wandb.log({'num_param/deform': deform_n_param})

    if model_args.use_ex_feature:
        refine_model = RefineModel(
            exp_dim=t_dim,
            feature_dim=gaussians.EX_FEATURE_DIM, 
            exp_multires=0, 
            out_rescale=model_args.refine_out_rescale,
            parser_type=model_args.refine_parser_type,
            pix2pix_n_blocks=model_args.pix2pix_n_blocks,
            stylegan_n_blocks=model_args.stylegan_n_blocks,
            stylegan_noise=model_args.stylegan_noise,
            )
        refine_model.train_setting(opt)
        refine_n_param = get_num_trainable_params(refine_model.network)
        model_summary += f"{'Refine'.ljust(20)}: {refine_n_param}\n"
        if WANDB_FOUND:
            wandb.log({'num_param/refine': refine_n_param})
    else:
        refine_model = None

    if model_args.layer_model != 'none':
        layer_model = LayerModel(
            model_args.layer_model,
            img_size=(scene.train_cameras[1.0][0].image_height, scene.train_cameras[1.0][0].image_width),
            exp_dim=t_dim,
            feature_dim=model_args.layer_feature_dim, 
            layer_parser_rescale=model_args.layer_out_rescale,
            parser_type=model_args.layer_parser_type,
            exp_multires=model_args.layer_input_exp_multires,
            t_multires=model_args.layer_input_t_multires,
            pose_multires=model_args.layer_input_pose_multires,
            layer_encoding=model_args.layer_encoding,
            )
        layer_model.train_setting(opt)

        bg_n_param = get_num_trainable_params(layer_model.bg_model)
        model_summary += f"{'Layer Background'.ljust(20)}: {bg_n_param}\n"
        if WANDB_FOUND:
            wandb.log({'num_param/layer_bg': bg_n_param})
        if layer_model.fg_model is not None:
            fg_n_param = get_num_trainable_params(layer_model.fg_model)
            model_summary += f"{'Layer Background'.ljust(20)}: {fg_n_param}\n"
            if WANDB_FOUND:
                wandb.log({'num_param/layer_fg': fg_n_param})
    else:
        layer_model = None

    gaussians.training_setup(opt)


    with (Path(model_args.model_path) / 'model_summary.txt').open('w') as f:
        f.write(model_summary)


    # bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    bg_color = [0,0,0]
    if scene.background is not None:
        scene_background = torch.from_numpy(scene.background).permute([2,0,1]) / 255.
        scene_background = scene_background.to('cuda')

        # if layer_model is not None:
        #     # initialize bg layer
        #     layer_model.bg_model. = scene_background

    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    with (Path(model_args.model_path) / 'point_number.txt').open('w') as f:
        f.write(f"\n")

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        view_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if model_args.load2gpu_on_the_fly:
            view_cam.load2device()
        fid = view_cam.fid
        exp = view_cam.exp

        d_sh = None
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            exp_input = exp.unsqueeze(0).expand(N, -1)

            # ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            ast_noise = 0
            d_xyz, d_rotation, d_scaling, d_sh = deform.step(gaussians.get_xyz.detach(), exp_input + ast_noise)
            if not model_args.deform_sh:
                d_sh = None

        # Render
        render_ex_feature = model_args.use_ex_feature and iteration >= opt.warm_up_cnn_refinement
        render_pkg_re = render(view_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, model_args.is_6dof, d_sh, render_ex_feature=render_ex_feature)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        alpha = render_pkg_re["alpha"]


        if model_args.use_ex_feature and iteration >= opt.warm_up_cnn_refinement:
            # apply feature + CNN based appearance refinement
            feature_im = render_pkg_re["feature_im"] # [h, w, EX_FEATURE_DIM]

            exp_input = exp.unsqueeze(0)
            time_input = view_cam.fid.unsqueeze(0)
            # time_input = torch.zeros_like(exp.unsqueeze(0))
            # ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(1, -1) * time_interval * smooth_term(iteration)

            refined_rgb = refine_model.step(feature_im, exp_input, time=time_input)[0]
            gs_img = image

            if model_args.refine_mode == 'add':
                image = torch.clamp(image + refined_rgb, 0, 1)
            elif model_args.refine_mode == 'replace':
                # assert opt.warm_up_cnn_refinement == 0
                
                if model_args.refine_parser_type == 'pix2pix':
                    refined_rgb = (refined_rgb + 1.) / 2.
                
                image = torch.clamp(refined_rgb, 0, 1)
            else:
                raise NotImplementedError
        else:
            gs_img = None
            refined_rgb = None

        # apply background
        fg = bg = None
        apply_layer = layer_model is not None and iteration >= opt.warm_up_layer
        if apply_layer:
            if gs_img is None:
                gs_img = image
            time_input = fid.unsqueeze(0)
            exp_input = exp.unsqueeze(0)
            cam_pose_input = torch.from_numpy(np.concatenate([view_cam.T, view_cam.look_at[:,0]])).type_as(exp_input).unsqueeze(0)

            bg = layer_model.get_background(time_input, exp_input, cam_pose_input)
            if layer_model.fg_model is not None:
                # blend foreground & background
                fg = layer_model.get_background(time_input, exp_input, cam_pose_input) # [4, H, W]
                image = fg[3:] * fg[:3] + (1.-fg[3:]) * alpha * image + (1.-fg[3:]) * (1.-alpha) * bg
            else:
                image = image + (1.-alpha) * bg
                # image = bg
                
        elif scene.background is not None:
            # composite with given background image
            # only do so when layer model is not used
            image = image + (1.-alpha) * scene_background
        else:
            # use black or white bacground
            if model_args.white_background:
                image = image + (1.-alpha) * 1.
        image = torch.clamp(image, 0, 1)

        # Loss
        log_dict = {}
        gt_image = view_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        log_dict['train/l1'] = Ll1.item()

        if opt.lambda_alpha_loss > 0 and apply_layer:
            assert view_cam.valid_mask is not None
            # the idea of alpha loss and valid mask is not to force Gaussian to have exactly alpha mask
            # but to prevent Gaussian to grow to unncessary place
            # we hence only penalize larger alpha
            valid_mask = torch.from_numpy(view_cam.valid_mask).type_as(alpha)
            alpha_loss = torch.where(alpha[0] > valid_mask, (alpha[0] - valid_mask)**2, 0).mean()
            log_dict['train/alpha_loss'] = alpha_loss.item()
        else:
            alpha_loss = 0


        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.lambda_alpha_loss * alpha_loss
        log_dict['train/total_loss'] = loss.item()
        loss.backward()

        iter_end.record()

        if model_args.load2gpu_on_the_fly:
            view_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            # cur_psnr = training_report(tb_writer, iteration, log_dict, l1_loss,
            #                            testing_iterations, scene, render, (pipe, background), deform,
            #                            model_args.load2gpu_on_the_fly, model_args.is_6dof, model_args.deform_sh, log_every=opt.log_every)
            log_dict['train/elapsed_time'] = iter_start.elapsed_time(iter_end)
            cur_psnr = training_report(iteration, log_dict, testing_iterations, scene, renderFunc=render,
                    renderArgs=(pipe, background), deform=deform, refine_model=refine_model, layer_model=layer_model, model_args=model_args, opt_args=opt)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                if model_args.use_ex_feature:
                    refine_model.save_weights(args.model_path, iteration)
                if layer_model is not None:
                    layer_model.save_weights(args.model_path, iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    print(f"new points: {gaussians._xyz.shape[0]}")
                    with (Path(model_args.model_path) / 'point_number.txt').open('a') as f:
                        f.write(f"Iter [{iteration}], Point: {gaussians._xyz.shape[0]}\n")

                if iteration % opt.opacity_reset_interval == 0 or (
                        model_args.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
                if refine_model is not None:
                    refine_model.optimizer.step()
                    refine_model.optimizer.zero_grad()
                    refine_model.update_learning_rate(iteration)
                if layer_model is not None:
                    layer_model.optimizer.step()
                    layer_model.optimizer.zero_grad()
                    layer_model.update_learning_rate(iteration)


            if iteration % args.log_im_every == 0:
                train_log_path = Path(model_args.model_path) / 'train_log'
                train_log_path.mkdir(exist_ok=True, parents=True)
                torchvision.utils.save_image(image, os.path.join(str(train_log_path), '{0:05d}'.format(iteration) + ".png"))
                if refined_rgb is not None:
                    torchvision.utils.save_image(gs_img, os.path.join(str(train_log_path), '{0:05d}_gs'.format(iteration) + ".png")) # result rendered by gaussian
                    torchvision.utils.save_image(refined_rgb, os.path.join(str(train_log_path), '{0:05d}_refine'.format(iteration) + ".png"))
                # torchvision.utils.save_image(feature_im[:3], os.path.join(str(train_log_path), '{0:05d}_feature'.format(iteration) + ".png"))
                if fg is not None:
                    torchvision.utils.save_image(fg, os.path.join(str(train_log_path), '{0:05d}_layer_fg'.format(iteration) + ".png")) 
                if bg is not None:
                    torchvision.utils.save_image(bg, os.path.join(str(train_log_path), '{0:05d}_layer_bg'.format(iteration) + ".png")) 

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args, opt_args, pipe_args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    with open(os.path.join(args.model_path, "opt_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt_args))))

    with open(os.path.join(args.model_path, "pipe_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(pipe_args))))

    if WANDB_FOUND:
        args_dict = args.__dict__
        args_dict.update(opt_args.__dict__)
        args_dict.update(pipe_args.__dict__)
        wandb.init(
            config=args_dict,
            project='gs_head_refine',
            group=Path(args.source_path).name,
            name=f'{Path(args.model_path).parent.name}/{Path(args.model_path).name}',
            mode=opt_args.wandb_mode,
            )
        
        # log wandb run id
        with open(os.path.join(args.model_path, "wandb_id.txt"), 'w') as f:
            f.write(wandb.run.id)
        

    # # Create Tensorboard writer
    # tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # return tb_writer


def training_report(iteration, log_dict, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, refine_model=None, layer_model=None, model_args=None, opt_args=None):
    # if tb_writer:
    #     tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
    #     tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
    #     tb_writer.add_scalar('iter_time', elapsed, iteration)

    if WANDB_FOUND and iteration % opt_args.log_every == 0:
        wandb.log(
            log_dict,
            step=iteration,
            )
        wandb.log(
            {
                'train/total_points': scene.gaussians.get_xyz.shape[0], 
            },
            step=iteration,
            )

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'val_iter', 'cameras': scene.getTestCameras()},
                              {'name': 'train_iter',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, view_cam in enumerate(config['cameras']):
                    if model_args.load2gpu_on_the_fly:
                        view_cam.load2device()
                    fid = view_cam.fid
                    exp = view_cam.exp
                    xyz = scene.gaussians.get_xyz
                    exp_input = exp.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling, d_sh = deform.step(xyz.detach(), exp_input)
                    if not model_args.deform_sh:
                        d_sh = None

                    render_ex_feature = model_args.use_ex_feature and iteration >= opt_args.warm_up_cnn_refinement
                    render_pkg_re = renderFunc(view_cam, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, model_args.is_6dof, d_sh, render_ex_feature=render_ex_feature)
                    image = render_pkg_re['render']
                    alpha = render_pkg_re['alpha']
                    # image = torch.clamp(
                    #     renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, model_args.is_6dof, d_sh)["render"],
                    #     0.0, 1.0)
                    

                    if model_args.use_ex_feature and iteration >= opt_args.warm_up_cnn_refinement:
                        # apply feature + CNN based appearance refinement
                        feature_im = render_pkg_re["feature_im"] # [h, w, EX_FEATURE_DIM]

                        exp_input = exp.unsqueeze(0)
                        time_input = view_cam.fid.unsqueeze(0)

                        refined_rgb = refine_model.step(feature_im, exp_input, time=time_input)[0]
                        gs_img = image

                        if model_args.refine_mode == 'add':
                            image = torch.clamp(image + refined_rgb, 0, 1)
                        elif model_args.refine_mode == 'replace':                            
                            if model_args.refine_parser_type == 'pix2pix':
                                refined_rgb = (refined_rgb + 1.) / 2.
                            image = torch.clamp(refined_rgb, 0, 1)
                        else:
                            raise NotImplementedError
                    else:
                        gs_img = None
                        refined_rgb = None

                    # apply background
                    fg = bg = None
                    if layer_model is not None and iteration >= opt_args.warm_up_layer:
                        if gs_img is None:
                            gs_img = image
                        time_input = fid.unsqueeze(0)
                        exp_input = exp.unsqueeze(0)
                        cam_pose_input = torch.from_numpy(np.concatenate([view_cam.T, view_cam.look_at[:,0]])).type_as(exp_input).unsqueeze(0)
                        bg = layer_model.get_background(time_input, exp_input, cam_pose_input)
                        if layer_model.fg_model is not None:
                            # blend foreground & background
                            fg = layer_model.get_background(time_input, exp_input, cam_pose_input) # [4, H, W]
                            image = fg[3:] * fg[:3] + (1.-fg[3:]) * alpha * image + (1.-fg[3:]) * (1.-alpha) * bg
                        else:
                            image = image + (1.-alpha) * bg
                            # image = bg
                            
                    elif scene.background is not None:
                        # composite with given background image
                        # only do so when layer model is not used
                        scene_background = torch.from_numpy(scene.background).permute([2,0,1]) / 255.
                        scene_background = scene_background.to('cuda')
                        image = image + (1.-alpha) * scene_background
                    else:
                        # use black or white bacground
                        if model_args.white_background:
                            image = image + (1.-alpha) * 1.
                    image = torch.clamp(image, 0, 1)

                    

                    image = torch.clamp(image, 0, 1)
                    gt_image = torch.clamp(view_cam.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if model_args.load2gpu_on_the_fly:
                        view_cam.load2device('cpu')
                    if idx < 5:
                        train_log_path = Path(model_args.model_path) / f"{config['name']}_log"
                        train_log_path.mkdir(exist_ok=True, parents=True)
                        torchvision.utils.save_image(image, str(train_log_path / f'{view_cam.image_name}_{iteration:05d}.png'))
                        if iteration == testing_iterations[0]:
                            torchvision.utils.save_image(gt_image, str(train_log_path / f'{view_cam.image_name}_{iteration:05d}_GT.png'))

                        if refined_rgb is not None:
                            torchvision.utils.save_image(gs_img, str(train_log_path / f'{view_cam.image_name}_{iteration:05d}_gs.png')) # result rendered by gaussian
                            torchvision.utils.save_image(refined_rgb, str(train_log_path / f'{view_cam.image_name}_{iteration:05d}_refine.png')) 
                        if fg is not None:
                            torchvision.utils.save_image(fg, str(train_log_path / f'{view_cam.image_name}_{iteration:05d}_layer_fg.png')) 
                        if bg is not None:
                            torchvision.utils.save_image(bg, str(train_log_path / f'{view_cam.image_name}_{iteration:05d}_layer_bg.png')) 
                        

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'val_iter' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if WANDB_FOUND:
                    wandb.log(
                        {
                            f"{config['name']}/l1": l1_test,
                            f"{config['name']}/psnr": psnr_test,
                        },
                        step=iteration,
                    )

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
                    

        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[500, 1000, 2000, 5000, 7000] + list(range(10000, 40001, 2000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--log_im_every', type=int, default=99999999)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
