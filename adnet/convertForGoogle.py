from unet256 import Unet256Model
from data.single_dataset import SingleDataset
import numpy as np
from PIL import Image
import torch
import tensorflow as tf
from onnx_tf.backend import prepare
import onnx

from torch.autograd  import Variable
dataset = SingleDataset('../datasets/SBUsd/Test/TestA/')
_data = None
for i,data in enumerate(dataset):
    _data=data
    break


import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

model = Unet256Model(load_model='135_net_D.pth')
model.print_net()

# Export the model
torch_out = torch.onnx.export(model.net,             # model being run
                               Variable(data['A'].cuda(0),requires_grad = 0),                       # model input (or a tuple for multiple inputs)
                               "model.onnx", # where to save the model (can be a file or file-like object)
                               input_names=['input'], output_names=['output'])




# Load ONNX model and convert to TensorFlow format
model_onnx = onnx.load('model.onnx')

tf_rep = prepare(model_onnx)

# Print out tensors and placeholders in model (helpful during inference in TensorFlow)
print(tf_rep.tensor_dict)

# Export model as .pb file
tf_rep.export_graph('modelTF.pb')

