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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import torchvision

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        # try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in sorted(os.listdir(test_dir)):
            if not method.startswith("ours"):
                continue
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []


            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

            if args.log_error_map:
                error_map_path = Path(renders_dir).parent / 'error'
                error_map_path.mkdir(exist_ok=True, parents=True)
                for idx in tqdm(range(len(renders)), desc="Error map saving progress"):
                    error_map = torch.clamp(torch.abs(renders[idx] - gts[idx]) / 0.3, 0., 1.)[0]
                    torchvision.utils.save_image(error_map, str(error_map_path / image_names[idx]))

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update(
                {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
            
            ssim_worst = np.argsort(torch.tensor(ssims).cpu().numpy())[:50]
            image_names_ = [image_names[i] for i in ssim_worst]
            per_view_dict[scene_dir][method].update(
                {"SSIM_worst_50": {name: ssim for ssim, name in zip(torch.tensor(ssims)[ssim_worst].tolist(), [image_names[i] for i in ssim_worst])}}
            )
            psnr_worst = np.argsort(torch.tensor(psnrs).cpu().numpy())[:50]
            per_view_dict[scene_dir][method].update(
                {"PSNR_worst_50": {name: psnr for psnr, name in zip(torch.tensor(psnrs)[psnr_worst].tolist(), [image_names[i] for i in psnr_worst])}}
            )
            lpips_worst = np.flip(np.argsort(torch.tensor(lpipss).cpu().numpy()))[:50].copy()
            per_view_dict[scene_dir][method].update(
                {"LPIPS_worst_50": {name: lpips for lpips, name in zip(torch.tensor(lpipss)[lpips_worst].tolist(),[image_names[i] for i in lpips_worst])}}
            )

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)
            

        mean_psnr = torch.tensor(psnrs).mean().item()
        mean_ssim = torch.tensor(ssims).mean().item()
        mean_lpips = torch.tensor(lpipss).mean().item()
            
        if WANDB_FOUND:
            wandb_id_file = Path(scene_dir) / 'wandb_id.txt'
            if wandb_id_file.exists():
                with wandb_id_file.open('r') as f:
                    wandb_id = f.read()
                wandb.init(
                    project='gs_head_refine',
                    id=wandb_id,
                    resume='must',
                )

                wandb.log({
                    'test/psnr': mean_psnr,
                    'test/ssim': mean_ssim,
                    'test/lpips': mean_lpips,
                })
            


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--log_error_map', action='store_true')
    args = parser.parse_args()
    evaluate(args.model_paths)
