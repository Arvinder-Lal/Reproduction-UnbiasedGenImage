from PIL import Image
import torch.utils.data as data
import os
import numpy as np
# import logging

# _logger = logging.getLogger(__name__) not used

def load_class_map(map_or_filename, root=''):
    if isinstance(map_or_filename, dict):
        assert dict, 'class_map dict must be non-empty'
        return map_or_filename
    class_map_path = map_or_filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, class_map_path)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % map_or_filename
    class_map_ext = os.path.splitext(map_or_filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    elif class_map_ext == '.pkl':
        with open(class_map_path,'rb') as f:
            class_to_idx = pickle.load(f)
    else:
        assert False, f'Unsupported class map file extension ({class_map_ext}).'
    return class_to_idx

import torch
import torch.fft
from torchvision.transforms.functional import to_tensor, to_pil_image
#hinzugefügt
class FFTTransform:
    def __init__(self, device='cuda'):
        self.device = device
        self.cache = {}

    def __call__(self, img):
        img_t = to_tensor(img).mean(dim=0, keepdim=True)  # RGB zu Graustufen
        img_t = img_t.to(self.device)

        f = torch.fft.fft2(img_t)
        fshift = torch.fft.fftshift(f)

        magnitude = torch.abs(fshift)
        log_magnitude = 20 * torch.log10(magnitude + 1e-6)
        log_magnitude = (log_magnitude - log_magnitude.min()) / \
                        (log_magnitude.max() - log_magnitude.min() + 1e-6)
        result = (log_magnitude * 255).type(torch.uint8)
        return to_pil_image(result.cpu().repeat(3, 1, 1))  # Grau zu RGB


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pywt
from torchvision.transforms.functional import to_tensor as TF_to_tensor
from torchvision.transforms.functional import to_pil_image as TF_to_pil_image
class WaveletTransform:
    def __init__(self, wavelet='haar', level=3, mode='symmetric', normalize=True):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.normalize = normalize

    def __call__(self, img):
        img_t = TF.to_tensor(img)
        result_channels = []

        for c in range(img_t.shape[0]):
            channel = img_t[c].numpy()

            coeffs = pywt.wavedec2(channel, wavelet=self.wavelet, level=self.level, mode=self.mode)

            coeff_arr, _ = pywt.coeffs_to_array(coeffs)

            if self.normalize:
                coeff_arr = np.log(np.abs(coeff_arr) + 1e-6)
                coeff_tensor = torch.from_numpy(coeff_arr).unsqueeze(0)
                coeff_tensor = F.interpolate(
                    coeff_tensor.unsqueeze(0),
                    size=(224, 224),
                    mode='bilinear'
                ).squeeze(0)

                result_channels.append(coeff_tensor)

                result = torch.cat(result_channels, dim=0)

        return TF.to_pil_image(result)

class ImageDataset(data.Dataset):

    def __init__(
            self,
            data,
            pre_transform,
            class_map=None,
            transform=None,
            target_transform=None,
            use_fft=False,  # Neuer Parameter (hinzugefügt)
            use_wavelet=False  # neuer Parameter (hinzugefügt)
    ): 
        #hinzugefügt
        self.use_fft = use_fft
        self.fft_transform = FFTTransform() if use_fft else None
        self.use_wavelet = use_wavelet
        self.wavelet_transform = WaveletTransform() if use_wavelet else None
        
        self.class_map = class_map
        self.data = data
        self.pre_transform = pre_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        #if index in self.cache and self.use_fft and self.fft_transform:
        #    return self.cache[index]
        
        img_path, target = self.data.iloc[index].path, self.data.iloc[index].target
        if self.class_map is None:
            target_encoded = 0 if target == "nature" else 1
        else:
            target_encoded = self.class_map[target]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise e

        if self.pre_transform is not None:
            img = self.pre_transform(img, target)

        #hinzugefügt
        if self.use_fft and self.fft_transform:
            img = self.fft_transform(img)
        if self.use_wavelet and self.wavelet_transform is not None:
            img = self.wavelet_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target_encoded

    def __len__(self):
        return len(self.data)


def create_dataset(data, pre_transform, transform, class_map):
    class_map = load_class_map(class_map)
    ds = ImageDataset(data, pre_transform, class_map=class_map, transform=transform)
    return ds
