import numpy as np
import torch
from torchvision.transforms import ToTensor, Lambda, Normalize, Compose

def get_radius(dist, nu):
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

def global_contrast_normalization(x: torch.tensor, scale='l2'):
    assert scale in ('l1', 'l2')
    n_features = int(np.prod(x.shape))
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features
    x /= x_scale
    return x

transform = Compose([
    ToTensor(),
    Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
    Normalize(0.5, 0.5)
])