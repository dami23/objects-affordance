import torch
import torch.nn as nn
import torchvision.models as res_model
from torch.autograd import Variable
from torchvision import transforms

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

from PIL import Image
import glob as gb
import numpy as np
import importlib
import time
import os

from option import Options

def img_to_variable(img):
    trans = transforms.Compose([transforms.ToTensor()])
    image = Image.open(img)
    width, height = image.size

    if width < 32 or height < 32:
        img = image.resize((width * 2, height * 2), Image.ANTIALIAS)
    else:
        img = image

    img_tensor = img.convert('RGB')
    img = trans(img_tensor)
    img = img.unsqueeze(0)
    img = Variable(img).cuda()

    return img

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

    checkpoint = torch.load('model/deepten_minc.pth')  # , map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(checkpoint['state_dict'])

    return model

def texture_extract(image, model):
    img = img_to_variable(image)

    out = model(img)
    a, b, c = outputs[0].size()
    texture_vec = outputs[0].view(a, b * c).data.cpu().unsqueeze(0).numpy()

    return texture_vec

if __name__ == '__main__':
    start_time = time.time()

    model = load_model()
    model.module.head[3].register_forward_hook(hook)

    save_path = "/informatik2/tams/home/mi/lieber_mi/projects/object_affordance/texture"
    try:
        os.mkdir(save_path)
    except OSError:
        pass

    img_files = gb.glob('/informatik2/tams/home/mi/lieber_mi/projects/object_affordance/RoI/val/clean/6/*.png')
    ln = len(img_files)

    index = 0
    counter = 0
    batch = 12

    while(counter < ln):
        print ("processing batch " + str(index + 1))

        imageList = []
        for i in range(counter, counter + batch - 1):
            imageList.append(img_files[i])

        for image in imageList:
            texture_vec = texture_extract(image, model)
            print (texture_vec.shape)
            np.save(os.path.join(save_path, os.path.splitext(os.path.basename(image))[0]), texture_vec)

            strPrint = 'complete %s feature extraction!' % os.path.splitext(os.path.basename(image))[0]
            print(strPrint)

        counter = counter + batch
        index = index + 1

        if counter % 24 == 0:
            del model
            model = load_model()
            model.module.head[3].register_forward_hook(hook)

            torch.cuda.empty_cache()

    stop_time = time.time() - start_time
    print('this code runs %s second.' % stop_time)
