# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline
import os

# import realesrgan.archs
# import realesrgan.data
# import realesrgan.models
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == '__main__':
    # root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)