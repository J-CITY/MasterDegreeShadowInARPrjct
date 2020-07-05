import sys
sys.path.insert(0, "./ADNET")
from utils import convert
import numpy as np
from PIL import Image
from torch.autograd  import Variable
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
          

            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            a = self.model(x)
            return torch.cat([x, a], 1)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(Generator, self).__init__()
        self.gpu_ids = gpu_ids

        unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class ADModel:
    net = None
    gpu = []

    def __init__(self, gpu_ids=[0],load_model=None, isCPU=False):
        self.isCPU = isCPU
        self.gpu = gpu_ids
        norm_layer = self.get_norm_layer('instance')
        self.net = Generator(3,1,8,64,use_dropout=False,norm_layer = norm_layer,gpu_ids = gpu_ids)#remove gpu_ids if convert to android
        if load_model is not None:
            self.net.load_state_dict(torch.load(load_model))
        if (not isCPU):
            self.net.cuda(0)

    def get_norm_layer(self, norm_type='instance'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,track_running_stats=True)
        elif layer_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError('layer [%s] is not found' % norm_type)
        return norm_layer

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('LR = %.7f' % lr)

    def print_net(self):
        networks.print_network(self.net)
    def test(self,data):
        if (not self.isCPU):
            return self.net.forward(Variable(data['A'].cuda(0),requires_grad = 0)).data
        else:
            return self.net.forward(Variable(data['A'],requires_grad = 0)).data


class ADNET:
    def __init__(self):
        self.model = ADModel(load_model='adnet/135_net_D.pth', isCPU=False)
        
    def getShadow(self, frame, width=256, height=256):
        outim = self.model.test(convert(frame))
        im_out = outim[0].cpu().float().numpy()
        im_out = np.transpose(im_out, (1,2,0))
        im_out = (im_out+1)/2*255
        im_out = im_out.astype('uint8')
    
        gray = Image.fromarray(np.squeeze(im_out, axis =2)).resize((int(width), int(height)))
        shadowFrame = np.array(gray)
        return shadowFrame
        