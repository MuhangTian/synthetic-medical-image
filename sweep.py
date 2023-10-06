import argparse
import time

import yaml

import wandb
from model.ddpm import GaussianDiffusion, Trainer, Unet


def sweep(args, run_id: str):
    sub_dir = "sweep"
    
    model = Unet(
        dim = args.dim,
        channels = args.channels,
        resnet_block_groups = args.resnet_block_groups,
        learned_sinusoidal_dim = args.learned_sinusoidal_dim,
        attn_dim_head = args.attn_dim_head,
        attn_heads = args.attn_heads,
        dim_mults = (1, 2, 4, 8),
        full_attn = (False, False, False, True),
        flash_attn = True,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = args.image_size,
        timesteps = args.timesteps,           # number of steps
        sampling_timesteps = args.sampling_timesteps,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        objective = args.objective,
        beta_schedule = args.beta_schedule,
        auto_normalize = True,
    )

    trainer = Trainer(
        diffusion,
        args.load_path,
        train_batch_size = args.batch_size,
        train_lr = args.lr,
        train_num_steps = args.num_steps,
        gradient_accumulate_every = args.gradient_accumulate_every,
        ema_decay = args.ema_decay,
        save_and_sample_every = args.save_and_sample_every,
        results_folder = f"./results/{sub_dir}/{run_id}",
        num_fid_samples=args.num_fid_samples,
        wandb = wandb,
        amp = True,        
        calculate_fid = True,
    )

    trainer.train()
    
if __name__ == "__main__":
    wandb.init()
    args = wandb.config
    sweep(args, run_id=run_id)
        