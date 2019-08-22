import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import BatchCollator
from util import postprocess_image


class Visualizer():
    def __init__(self, cfg, dataloader, model, device):
        self.cfg = cfg
        self.image_height = cfg.image_height
        self.image_width = cfg.image_width
        self.image_channel_size = cfg.image_channel_size

        self.num_dataloaders = cfg.num_dataloaders
        self.device = device

        self.batch_size = cfg.batch_size
        self.num_instances = cfg.num_instances
        self.num_memories = cfg.num_memories

        self.condi_loss_coef = cfg.condi_loss_coef

        self.model = model

        if cfg.test_set == 'train':
            self.test_set = dataloader.train_dataset
        else:
            self.test_set = dataloader.test_dataset

        self.collator = BatchCollator(image_height=self.image_height,
                                      image_width=self.image_width,
                                      image_channel_size=self.image_channel_size,
                                     )

    def visualize(self):
        self.testloader = DataLoader(dataset=self.test_set,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     collate_fn=self.collator,
                                     num_workers=self.num_dataloaders,
                                    )
        for i, batch in enumerate(self.testloader):
            self.model.eval()

            batch = [b.to(self.device) for b in batch]
            imgs, labels, instances = batch[0], batch[1], batch[2]
            batch_size = imgs.size(0)

            with torch.no_grad():
                result = self.model(imgs)

            rec_imgs = result['rec_x']
            cls_logit = result['logit_x']

            cls_pred = cls_logit.max(1)[1]
            pred_imgs = self.testloader.dataset[cls_pred.item()][0]

            imgs = postprocess_image(imgs)
            rec_imgs = postprocess_image(rec_imgs)
            pred_imgs = postprocess_image(pred_imgs.unsqueeze(0))

            if self.condi_loss_coef != 0.0:
                idx = instances
            else:
                idx = torch.randint(self.num_memories, (1,))
            with torch.no_grad():
                result = self.model.generate_from_memory(idx)

            rec_gen_imgs = result['rec_x']
            cls_gen_logit = result['logit_x']

            cls_gen_pred = cls_gen_logit.max(1)[1]
            pred_gen_imgs = self.testloader.dataset[cls_gen_pred.item()][0]

            rec_gen_imgs = postprocess_image(rec_gen_imgs)
            pred_gen_imgs = postprocess_image(pred_gen_imgs.unsqueeze(0))

            figure, axarr = plt.subplots(2, 3, figsize=(8, 5))
            axarr[0][0].set_title('O. image, Idx: %i' % (instances.item()))
            axarr[0][0].imshow(np.squeeze(imgs), cmap='gray')
            axarr[0][0].axis('off')

            axarr[0][1].set_title('R. image')
            axarr[0][1].imshow(np.squeeze(rec_imgs), cmap='gray')
            axarr[0][1].axis('off')

            axarr[0][2].set_title('P. image, Idx: %i' % (cls_pred.item()))
            axarr[0][2].imshow(np.squeeze(pred_imgs), cmap='gray')
            axarr[0][2].axis('off')

            axarr[1][0].axis('off')

            axarr[1][1].set_title('G. image')
            axarr[1][1].imshow(np.squeeze(rec_gen_imgs), cmap='gray')
            axarr[1][1].axis('off')

            axarr[1][2].set_title('GP. image, Idx: %i' % (cls_gen_pred.item()))
            axarr[1][2].imshow(np.squeeze(pred_gen_imgs), cmap='gray')
            axarr[1][2].axis('off')

            plt.show(block=False)
            print()
            input('Type enter key to watch next result')
            print()
            plt.close(figure)
