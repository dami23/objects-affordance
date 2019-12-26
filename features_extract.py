import torch
import torch.nn as nn

import importlib
from option import Options

__all__ = ['visual_extract', 'load_model', 'texture_extract']

def visual_extract(image, net):
    net_layer4 = nn.Sequential(*list(net.children())[:8])
    fm_layer4 = net_layer4(image)
    fm_layer4_3d = fm_layer4.data.cpu()

    return fm_layer4_3d

outputs = []
def hook(module, input, output):
    outputs.append(output)

def load_model():
    device_ids = [0, 1]

    args = Options().parse()
    models = importlib.import_module('model.' + 'deepten')
    model = models.Net(args)
    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids)

    checkpoint = torch.load('model/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    return model

def texture_extract(image, model):
    out = model(image)

    a, b, c = outputs[0].size()
    texture_vec = outputs[0].view(a, b * c).data.cpu().unsqueeze(0)

    return texture_vec
