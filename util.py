import numpy as np
import torch


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if imgs.size(1) == 1:
            tensor = imgs.mul_(self.std[0]).add_(self.mean[0])
        else:
            tensor = []
            for img in imgs:
                for t, m, s in zip(tensor, self.mean, self.std):
                    t.mul_(s).add_(m)
            tensor = torch.stack(tensor)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def postprocess_image(imgs):
    imgs = imgs.squeeze(0)
    imgs = imgs.cpu().numpy()
    imgs = np.transpose(imgs, [1, 2, 0])
    return imgs
