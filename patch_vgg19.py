import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np
import glob as gb
import os
import time
import cv2

start_time = time.time()

def img_to_variable(img):
    image = Image.fromarray(img)
    img = image.resize((224, 224), Image.ANTIALIAS).convert('RGB')
    img_tensor = img.convert('RGB')

    trans = transforms.Compose([transforms.ToTensor()])
    img = trans(img_tensor)
    img = img.unsqueeze(0)
    img = Variable(img).cuda()

    return img

def visual_extract(image, net):
    img = img_to_variable(image)

    net_layer4 = nn.Sequential(*list(net.features)[:37])
    fm_layer4 = net_layer4(img)
    fm_layer4_3d = fm_layer4.data.cpu().numpy()

    return fm_layer4_3d

if __name__ == '__main__':
    net = models.vgg19(pretrained=True)
    if torch.cuda.is_available():
        net = net.cuda()
    for param in net.parameters():
        param.requires_grad = False

    label_path = '/informatik2/tams/home/mi/lieber_mi/projects/dataset/mydataset/labels/val/'
    img_path = '/informatik2/tams/home/mi/lieber_mi/projects/dataset/mydataset/val/'
    for root, dirs, files in os.walk(img_path):
       for dir in dirs:
           images = os.listdir(img_path + dir)
           for image in images:
               img_name = os.path.basename(image)
               fname = os.path.splitext(img_name)[0]
               srcImg = cv2.imread(os.path.join(img_path + dir, img_name))

               label_txt = os.path.join(label_path + dir, fname + '.txt')
               label_file = open(label_txt)
               data_line = label_file.readlines()

               regionList = []
               fm_concat = []
               for line in data_line:
                   data_list = line.split()

                   lr = int(data_list[1])
                   lc = int(data_list[2])
                   rr = int(data_list[3])
                   rc = int(data_list[4])

                   roi = srcImg[lc:rc, lr:rr]
                   regionList.append(roi)

               for region in regionList:
                   vis_fea = visual_extract(region, net)

                   save_path = "/informatik2/tams/home/mi/lieber_mi/projects/object_affordance/vis_tex/val/" + dir
                   try:
                       os.mkdir(save_path)
                   except OSError:
                       pass

                   if len(data_line) == 1:
                       np.save(os.path.join(save_path, os.path.splitext(os.path.basename(image))[0]), vis_fea)
                   elif len(data_line) > 1:
                       for i in range(len(data_line)):
                           np.save(os.path.join(save_path, os.path.splitext(os.path.basename(image))[0] + '_' + str(i)), vis_fea)

               print ('complete %s feature extraction!' % img_name)

    stopped = time.time() - start_time
    print ('this code runs %s second.' % stopped)
