import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode
import torch
import torch.nn.functional as F
import PIL
import os
import pandas as pd
from typing import Optional, Dict, List, Tuple

ImageFile.LOAD_TRUNCATED_IMAGES = True

# hinzugefügt
from .get_data import get_data
from .get_transform import get_transform


# hinzugefügt
class FFTTransform:
    def __init__(self, log_scale=True):
        self.log_scale = log_scale

    def __call__(self, img):
        img_gray = img.convert("L")
        img_np = np.array(img_gray)
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        if self.log_scale:
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        magnitude_spectrum -= magnitude_spectrum.min()
        magnitude_spectrum /= magnitude_spectrum.max()
        magnitude_spectrum = (magnitude_spectrum * 255).astype(np.uint8)

        return Image.fromarray(magnitude_spectrum).convert("RGB")


import pywt
import torch
class OldWaveletTransform:
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

import pywt
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image


class WaveletTransform:
    def __init__(self, wavelet='haar', level=3, apply_log=True):
        self.wavelet = wavelet
        self.level = level
        self.apply_log = apply_log

    def __call__(self, img):
        img_tensor = to_tensor(img)  # [3, H, W]
        wavelet_channels = []

        for c in range(img_tensor.shape[0]):
            channel_np = img_tensor[c].numpy()
            coeffs = pywt.wavedec2(channel_np, self.wavelet, level=self.level)

            LL = coeffs[0]
            details = coeffs[1:]

            # Calculate output dimensions based on LL shape
            out_h, out_w = LL.shape[0], LL.shape[1]

            # Create combined array with correct dimensions
            combined = np.zeros((out_h * (self.level + 1), out_w * 3))  # *3 for H,V,D components

            # Place LL coefficients (no longer scaling by 2)
            combined[0:out_h, 0:out_w] = np.interp(LL, (LL.min(), LL.max()), (0, 1))

            for i, detail in enumerate(details):
                cH, cV, cD = detail
                base_y = out_h * (i + 1)

                # Normalize each component
                cH = np.interp(cH, (cH.min(), cH.max()), (0, 1))
                cV = np.interp(cV, (cV.min(), cV.max()), (0, 1))
                cD = np.interp(cD, (cD.min(), cD.max()), (0, 1))

                combined[base_y:base_y + out_h, 0:out_w] = cH
                combined[base_y:base_y + out_h, out_w:2 * out_w] = cV
                combined[base_y:base_y + out_h, 2 * out_w:3 * out_w] = cD

            if self.apply_log:
                combined = np.log(combined + 1e-6)

            wavelet_channels.append(combined)

        max_h = max(ch.shape[0] for ch in wavelet_channels)
        max_w = max(ch.shape[1] for ch in wavelet_channels)

        stacked = np.stack([
            np.resize(ch, (max_h, max_w)) for ch in wavelet_channels
        ], axis=-1)

        stacked = (stacked - stacked.min()) / (stacked.max() - stacked.min() + 1e-9)
        stacked = (stacked * 255).astype(np.uint8)

        return Image.fromarray(stacked)


def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if hasattr(opt, 'csv_data_path') and opt.csv_data_path:
        if opt.is_validating:
            data_val, paths = get_data(opt, is_training=False,
                                       balance_val_classes=opt.balance_val_classes,
                                       balance_train_classes=opt.balance_train_classes)
        else:
            data_train, data_val, paths = get_data(opt, is_training=True,
                                                   balance_val_classes=opt.balance_val_classes,
                                                   balance_train_classes=opt.balance_train_classes)

        if opt.isTrain:
            dset = CustomImageDataset(data_train, opt)
            print("Build dset with data_train")
        else:
            dset = CustomImageDataset(data_val, opt)
            print("Build dset with data_val")
        print(f"the dataset len is: {len(dset)}")
        return dset, paths

    # Original ImageFolder functionality
    transform_list = get_base_transforms(opt)

    if hasattr(opt, 'use_fft') and opt.use_fft:
        print("Using FFT transform")
        transform_list.insert(0, FFTTransform())

    if hasattr(opt, 'use_wavelet') and opt.use_wavelet:
        print("Using Wavelet transform")
        transform_list.insert(0, WaveletTransform())

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dset = SafeImageFolder(
        root,
        transform=transforms.Compose(transform_list),
        is_valid_file=is_valid_file_func
    )

    dset.classes = ["nature", "ai"]
    dset.class_to_idx = {"nature": 0, "ai": 1}

    print(f"the dataset len is: {len(dset)}")
    paths = [sample[0] for sample in dset.samples]
    return dset, paths


def get_base_transforms(opt):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)

    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))

    if opt.is_aug:
        print("aug in train")
        aug_func = transforms.Lambda(lambda img: data_augment(img, opt))
    else:
        print("NO aug")
        aug_func = transforms.Lambda(lambda img: img)

    return [rz_func, crop_func, flip_func, aug_func]


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, opt, class_names=None):
        self.df = df
        self.opt = opt
        self.class_names = class_names or ["nature", "ai"]
        self.class_to_idx = {"nature": 0, "ai": 1}

        transform_list = get_base_transforms(opt)

        if hasattr(opt, 'use_fft') and opt.use_fft:
            print("Using FFT transform")
            transform_list.insert(0, FFTTransform())

        if hasattr(opt, 'use_wavelet') and opt.use_wavelet:
            print("Using Wavelet transform")
            transform_list.insert(0, WaveletTransform())

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        composed_post_transform = transforms.Compose(transform_list)
        pil_transform_fn = get_transform(opt, df)

        def full_transform(img, label):
            img = pil_transform_fn(img, label)
            if hasattr(opt, 'use_fft') and opt.use_fft:
                img = FFTTransform()(img)
            if hasattr(opt, 'use_wavelet') and opt.use_wavelet:
                img = WaveletTransform()(img)
            img = composed_post_transform(img)
            return img

        self.transform = full_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        target = row['target']

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Skipping corrupted file: {img_path} (Error: {str(e)})")
            img = Image.new("RGB", (self.opt.loadSize, self.opt.loadSize), (0, 0, 0))
            target = "nature"

        target_encoded = self.class_to_idx[target]
        img = self.transform(img, target)
        return img, target_encoded


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

        # hinzugefügt
    if opt.jpeg96 or opt.size_constrained:
        opt.blur_prob = 0
        opt.jpg_prob = 0

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}


def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize, opt.loadSize), interpolation=rz_dict[interp])
