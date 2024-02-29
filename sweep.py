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
import subprocess,sys

import wandb
import shutil
from pathlib import Path

PROJECT = 'gs_head_refine'
SWEEP_ID = 'pozanwbq'

def sys_cmd(cmd):
    print(cmd)
    os.system(cmd) 

def train_agent(config=None):
    with wandb.init(config=config):
        args = wandb.config

        # modify the sweep configs
        args.test_iterations = -1
        args.save_iterations = -1
        args.model_path = f"sweep/{SWEEP_ID}/{wandb.run.id}"

        # invoke training
        cmd = "python train.py -s /home/tw554/Deformable-3D-Gaussians/data/GaussianHead/id1 --eval"
        for k in args.keys():
            if isinstance(args[k], bool):
                if args[k]:
                    cmd += f" --{k}"
            else:
                cmd += f" --{k} {args[k]}"
        sys_cmd(cmd)


        # print("Optimizing " + args.model_path)

        # # Initialize system state (RNG)
        # safe_state(args.quiet)

        # # Start GUI server, configure and run training
        # # network_gui.init(args.ip, args.port)
        # torch.autograd.set_detect_anomaly(args.detect_anomaly)
        # parser = ArgumentParser(description="Training script parameters")
        # lp = ModelParams(parser)
        # op = OptimizationParams(parser)
        # pp = PipelineParams(parser)

        # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)
        # # All done
        # print("\nTraining complete.")

        # invoke rendering
        # cmd = f"python render.py -m {args.model_path} --no_vid --no_extra"
        cmd = f"python render.py -m {args.model_path} --no_extra"
        sys_cmd(cmd)

        # invoke metrics
        cmd = f"python metrics.py -m {args.model_path}"
        sys_cmd(cmd)

        # clean up logged files
        shutil.rmtree(str(Path(args.model_path) / 'test' / 'ours_40000' / 'gt'))



if __name__ == "__main__":
    wandb.agent(f'{PROJECT}/{SWEEP_ID}', train_agent, count=18) 