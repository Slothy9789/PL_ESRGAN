import cv2
import math
import numpy as np
import os
import os.path as osp
import random

import torch
from torch.utils import data as data

from basicsr.data.degradations import random_mixed_kernels

from basicsr.utils import FileClient, img2tensor,imfrombytes_1_channels
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(suffix='basicsr')
class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(' ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        # blur settings for the first degradation
        # self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.use_hflip = opt['use_hflip']
        self.use_rot = opt['use_rot']
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        gt_path = self.paths[index]
        if 'TCO_MAP' in gt_path or 'le_' in gt_path or 're_' in gt_path:
            data_label = 'SELENE'
        elif 'MurrayLab_CTX' in gt_path or 'psp_' in gt_path or 'esp_' in gt_path:
            data_label = 'CTX'
        elif 'mdis_rtm' in gt_path or 'mercury_wac' in gt_path:
            data_label = 'Mercury'
        else:
            print('非CTX，非SELENE, 非Mercury')

        img_gt = imfrombytes_1_channels(gt_path)

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)

        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        if np.ndim(img_gt) == 2:
            img_gt = img_gt[:,:, np.newaxis]
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)

        return_d = {'gt': img_gt, 'kernel': kernel, 'gt_path': gt_path, 'data_label': data_label, 'use_hflip': self.use_hflip, 'use_rot': self.use_rot}
        return return_d

    def __len__(self):
        return len(self.paths)
