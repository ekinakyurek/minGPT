import math
import os
from argparse import ArgumentParser
import arrow
import numpy as np
import torch
import pathlib
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from mingpt.callback import CUDACallback
from mingpt.lr_decay import LearningRateDecayCallback
from mingpt.model import GPT
from pathlib import Path


if __name__ == '__main__':
    import os
    seed_everything(42)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print("loading model checkpoint")
    chk_path = "./checkpoints"
    chk_pts = sorted(Path(chk_path).glob("*.ckpt"))
    print("all available checkpoints: ", chk_pts)
    model = GPT.load_from_checkpoint(checkpoint_path=f'{chk_pts[-1]}/')
    print("finished loading monitor. LR: ", model.learning_rate)
    