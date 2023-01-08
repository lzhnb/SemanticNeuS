# Copyright (c) Zhihao Liang. All rights reserved.
import argparse
import os

import omegaconf
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import semneus

# set the random seed
semneus.set_random_seed(123456)


def get_args() -> argparse.Namespace:
    parser = semneus.get_default_args()

    # training parameters
    parser.add_argument("--batch_size", type=int, default=5000, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate or mlp")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # prase args
    args = get_args()
    conf = semneus.merge_config_file(args)
    print(f"Config:\n{omegaconf.OmegaConf.to_yaml(conf)}")

    # init train directory summary writer
    os.makedirs(conf.train_dir, exist_ok=True)
    summary_writer = SummaryWriter(conf.train_dir)

    # backup code and config
    proj_prefix = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), "..", x))
    semneus.backup(
        os.path.join(conf.train_dir, "backup"),
        [
            proj_prefix("semneus"),
            proj_prefix("cpp"),
            proj_prefix("scripts"),
            proj_prefix("setup.py"),
            proj_prefix("CMakeLists.txt"),
            conf.config,
        ],
    )

    # define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sem_neus = semneus.SemanticNeuS(**conf.model).to(device)
    sem_neus.train()
    optimizer = optim.Adam(sem_neus.parameters(), lr=conf.lr)
