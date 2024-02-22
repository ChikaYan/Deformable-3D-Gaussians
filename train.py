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
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
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


def training(model_args, opt, pipe, testing_iterations, saving_iterations):
    prepare_output_and_logger(model_args, opt, pipe)
    tb_writer = None
    gaussians = GaussianModel(model_args.sh_degree, model_args.use_ex_feature)

    scene = Scene(model_args, gaussians)

    t_dim = scene.train_cameras[1.0][0].exp.shape[-1]
    deform = DeformModel(t_dim=t_dim, is_blender=model_args.is_blender, is_6dof=model_args.is_6dof)
    deform.train_setting(opt)

    if model_args.use_ex_feature:
        refine_model = RefineModel(
            t_dim=t_dim,
            feature_dim=gaussians.EX_FEATURE_DIM, 
            t_multires=0, 
            cnn_out_rescale=model_args.cnn_out_rescale,
            parser_type=model_args.refine_parser_type,
            )
        refine_model.train_setting(opt)
    else:
        refine_model = None

    if model_args.layer_model != 'none':
        layer_model = LayerModel(
            model_args.layer_model,
            img_size=(scene.train_cameras[1.0][0].image_height, scene.train_cameras[1.0][0].image_width),
            t_dim=t_dim,
            feature_dim=model_args.layer_feature_dim, 
            t_multires=0, 
            layer_parser_rescale=model_args.layer_out_rescale,
            parser_type=model_args.layer_parser_type,
            )
        layer_model.train_setting(opt)
    else:
        layer_model = None

    gaussians.training_setup(opt)

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
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, model_args.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if model_args.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
        exp = viewpoint_cam.exp

        d_sh = None
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = exp.unsqueeze(0).expand(N, -1)

            # ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            ast_noise = 0
            d_xyz, d_rotation, d_scaling, d_sh = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
            if not model_args.deform_sh:
                d_sh = None

        # Render
        render_ex_feature = model_args.use_ex_feature and iteration >= opt.warm_up_cnn_refinement
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, model_args.is_6dof, d_sh, render_ex_feature=render_ex_feature)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        alpha = render_pkg_re["alpha"]


        if model_args.use_ex_feature and iteration >= opt.warm_up_cnn_refinement:
            # apply feature + CNN based appearance refinement
            feature_im = render_pkg_re["feature_im"] # [h, w, EX_FEATURE_DIM]

            time_input = exp.unsqueeze(0)
            # time_input = torch.zeros_like(exp.unsqueeze(0))
            # ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(1, -1) * time_interval * smooth_term(iteration)

            refined_rgb = refine_model.step(feature_im, time_input)[0]
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

        fg = bg = None
        if layer_model is not None and iteration >= opt.warm_up_layer:
            if gs_img is None:
                gs_img = image
            time_input = exp.unsqueeze(0)
            bg = layer_model.get_background(time_input)
            if layer_model.fg_model is not None:
                # blend foreground & background
                fg = layer_model.get_foreground(time_input) # [4, H, W]
                image = fg[3:] * fg[:3] + (1.-fg[3:]) * alpha * image + (1.-fg[3:]) * (1.-alpha) * bg
            else:
                image = image + (1.-alpha) * bg
                
        elif scene.background is not None:
            # composite with given background image
            # only do so when layer model is not used
            image = image + (1.-alpha) * scene_background
            

        # Loss
        log_dict = {}
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        log_dict['train/l1'] = Ll1.item()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        log_dict['train/total_loss'] = loss.item()
        loss.backward()

        iter_end.record()

        if model_args.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

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
        pass
        args_dict = args.__dict__
        args_dict.update(opt_args.__dict__)
        args_dict.update(pipe_args.__dict__)
        wandb.init(
            config=args_dict,
            project='gs_head_refine',
            group=Path(args.source_path).name,
            name=Path(args.model_path).name,
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
                for idx, viewpoint in enumerate(config['cameras']):
                    if model_args.load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    exp = viewpoint.exp
                    xyz = scene.gaussians.get_xyz
                    time_input = exp.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling, d_sh = deform.step(xyz.detach(), time_input)
                    if not model_args.deform_sh:
                        d_sh = None

                    render_ex_feature = model_args.use_ex_feature and iteration >= opt_args.warm_up_cnn_refinement
                    render_pkg_re = renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, model_args.is_6dof, d_sh, render_ex_feature=render_ex_feature)
                    image = render_pkg_re['render']
                    alpha = render_pkg_re['alpha']
                    # image = torch.clamp(
                    #     renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, model_args.is_6dof, d_sh)["render"],
                    #     0.0, 1.0)
                    

                    if model_args.use_ex_feature and iteration >= opt_args.warm_up_cnn_refinement:
                        # apply feature + CNN based appearance refinement
                        feature_im = render_pkg_re["feature_im"] # [h, w, EX_FEATURE_DIM]

                        time_input = exp.unsqueeze(0)
                        refined_rgb = refine_model.step(feature_im, time_input)[0]
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

                    fg = bg = None
                    if layer_model is not None and iteration >= opt_args.warm_up_layer:
                        if gs_img is None:
                            gs_img = image
                        time_input = exp.unsqueeze(0)
                        bg = layer_model.get_background(time_input)
                        if layer_model.fg_model is not None:
                            # blend foreground & background
                            fg = layer_model.get_foreground(time_input) # [4, H, W]
                            image = fg[3:] * fg[:3] + (1.-fg[3:]) * alpha * image + (1.-fg[3:]) * (1.-alpha) * bg
                        else:
                            image = image + (1.-alpha) * bg
                            
                    elif scene.background is not None:
                        # composite with given background image
                        # only do so when layer model is not used
                        scene_background = torch.from_numpy(scene.background).permute([2,0,1]) / 255.
                        scene_background = scene_background.to('cuda')
                        image = image + (1.-alpha) * scene_background
                    

                    image = torch.clamp(image, 0, 1)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if model_args.load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if idx < 5:
                        train_log_path = Path(model_args.model_path) / f"{config['name']}_log"
                        train_log_path.mkdir(exist_ok=True, parents=True)
                        torchvision.utils.save_image(image, str(train_log_path / f'{viewpoint.image_name}_{iteration:05d}.png'))
                        if iteration == testing_iterations[0]:
                            torchvision.utils.save_image(gt_image, str(train_log_path / f'{viewpoint.image_name}_{iteration:05d}_GT.png'))

                        if refined_rgb is not None:
                            torchvision.utils.save_image(gs_img, str(train_log_path / f'{viewpoint.image_name}_{iteration:05d}_gs.png')) # result rendered by gaussian
                            torchvision.utils.save_image(refined_rgb, str(train_log_path / f'{viewpoint.image_name}_{iteration:05d}_refine.png')) 
                        if fg is not None:
                            torchvision.utils.save_image(fg, str(train_log_path / f'{viewpoint.image_name}_{iteration:05d}_layer_fg.png')) 
                        if bg is not None:
                            torchvision.utils.save_image(bg, str(train_log_path / f'{viewpoint.image_name}_{iteration:05d}_layer_bg.png')) 
                        

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
                        default=[500, 1000, 5000, 7000] + list(range(10000, 40001, 2000)))
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
