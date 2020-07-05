import torch
from unet256 import Unet256Model
from data.single_dataset import SingleDataset
import numpy as np
from PIL import Image
import cv2
import torch

model = Unet256Model(load_model='135_net_D.pth', isCPU=True)

def main():
    model.net.eval()
    #dummy_input = torch.zeros(1, 3, 256, 256)
    dummy_input = torch.randn(1, 3, 256, 256)
    print(dummy_input.shape)
    torch.onnx.export(model.net, dummy_input, 'onnx_model.onnx', verbose=True)


if __name__ == '__main__':
    main()