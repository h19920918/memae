import argparse
from datetime import datetime
import multiprocessing as mp
import os
from pprint import pprint

import numpy as np
import random
import torch
from torch import nn
from torch.optim import Adam

from dataset import MNIST_Dataset
from model import ICVAE
from train import Trainer
from test import Tester
from visualize import Visualizer


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='data/MNIST')
    parser.add_argument('--prepro-dir', type=str, default='prepro/MNIST')

    parser.add_argument('--num-instances', type=int, default=60000)
    parser.add_argument('--num-classes', type=int, default=60000)
    parser.add_argument('--num-memories', type=int, default=100)
    parser.add_argument('--image-height', type=int, default=28)
    parser.add_argument('--image-width', type=int, default=28)
    parser.add_argument('--image-channel-size', type=int, default=1)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--log-dir', type=str, default='./logs-test/%s' % datetime.now().strftime('%b%d_%H-%M-%S'))
    parser.add_argument('--test-set', type=str, default='train')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--num-dataloaders', type=int, default=mp.cpu_count())

    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--cls-loss-coef', type=float, default=0.0)
    parser.add_argument('--entropy-loss-coef', type=float, default=0.0002)
    parser.add_argument('--condi-loss-coef', type=float, default=0.0)
    parser.add_argument('--addressing', type=str, default='soft')

    parser.add_argument('--conv-channel-size', type=int, default=16)
    parser.add_argument('--drop-rate', type=float, default=0.2)

    cfg = parser.parse_args()
    return cfg



def main(cfg):
    if not torch.cuda.is_available():
        print('CPU mode is not supported')
        exit(1)
    device = torch.device('cuda:0')

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if cfg.ckpt:
        if not os.path.exists(cfg.ckpt):
            print('Invalid ckpt path -->', cfg.ckpt)
            exit(1)
        ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc:storage)

        print(cfg.ckpt, 'loaded')
        loaded_cfg = ckpt['cfg'].__dict__
        pprint(loaded_cfg)
        del loaded_cfg['train']
        del loaded_cfg['test']
        del loaded_cfg['visualize']
        del loaded_cfg['batch_size']
        del loaded_cfg['ckpt']

        cfg.__dict__.update(loaded_cfg)
        print()
        print('Merged Config')
        pprint(cfg.__dict__)
        print()

        step = ckpt['step']
    else:
        os.makedirs(os.path.join(cfg.log_dir, 'ckpt'))
        step = 0

    dataloader = MNIST_Dataset(cfg=cfg,)
    model = ICVAE(cfg=cfg, device=device,)
    print()
    print(model)
    print()

    if torch.cuda.device_count() > 1 and cfg.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = Adam(params=model.parameters(),
                     lr=cfg.lr,
                     # weight_decay=cfg.weight_decay,
                    )

    if cfg.ckpt is not None:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    if cfg.train:
        trainer = Trainer(cfg=cfg,
                          dataloader=dataloader,
                          model=model,
                          optimizer=optimizer,
                          device=device,
                          step=step,
                         )
        trainer.train()
    elif cfg.test:
        tester = Tester(cfg=cfg,
                        dataloader=dataloader,
                        model=model,
                        device=device,
                       )
        tester.test()
    elif cfg.visualize:
        cfg.batch_size = 1
        visualizer = Visualizer(cfg=cfg,
                                dataloader=dataloader,
                                model=model,
                                device=device,
                               )
        visualizer.visualize()
    else:
        print('Select mode')
        exit(1)


if __name__ == '__main__':
    cfg = config()
    print('Config')
    pprint(cfg.__dict__)
    print()
    main(cfg)
