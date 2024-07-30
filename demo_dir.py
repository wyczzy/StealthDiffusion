import argparse
import os
import csv
import torch
from torch import nn
import timm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn import metrics

from networks.resnet import resnet50
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from loggers import Logger
from PIL import Image
import torchvision.transforms.functional as TF



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', nargs='+', type=str, default=["/path/to/val",
                                                                ])
parser.add_argument('-m','--model_path', type=str, default='/path/to/***.pth')
parser.add_argument('-p','--log_path', type=str, default='/path/to/logs')
parser.add_argument('-de','--device', type=str, default='cuda:1')
parser.add_argument('-a','--arch', type=str, default='resnet50')
parser.add_argument('-b','--batch_size', type=int, default=16)
parser.add_argument('-j','--workers', type=int, default=0, help='number of workers')
parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
parser.add_argument('-s', '--size', type=int, default=224, help='size')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
parser.add_argument('--size_only', action='store_true', help='only look at sizes of images in dataset')

opt = parser.parse_args()
p = opt.log_path
os.makedirs(p, exist_ok=True)
logger = Logger(name='demofiles', log_path=os.path.join(p,'{}.log'.format(opt.arch)))

dic = {
    "resnet50": "./checkpoints/resnet50.pth",
    "efficientnet-b0": "./checkpoints/efficientnet-b0.pth",
    "deit": "./checkpoints/deit.pth",
    "swin-t": "./checkpoints/swin-t.pth",
}


if(not opt.size_only):
  if(opt.arch=="resnet50"):
    model = resnet50(num_classes=1)
  elif(opt.arch=="efficientnet-b0"):
    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=1, image_size=None, )
  elif(opt.arch=="deit"):
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    model.head = torch.nn.Linear(in_features=768, out_features=1, bias=True)
  elif(opt.arch=="swin-t"):
      model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
      model.head = torch.nn.Linear(in_features=1024, out_features=1, bias=True)
  if(opt.model_path is not None):
      opt.model_path = dic[opt.arch]
      state_dict = torch.load(opt.model_path, map_location='cpu')
      from collections import OrderedDict

      new_state_dict = OrderedDict()
      if opt.arch == "clip":
          for k, v in state_dict.items():
              new_state_dict[k[7:]] = v
          state_dict = new_state_dict
  if("model" in state_dict.keys()):
    model.load_state_dict(state_dict['model'])
  else:
    model.load_state_dict(state_dict)
  model.eval()
  if(not opt.use_cpu):
      model.to(opt.device)

rz_func = transforms.Lambda(lambda img: custom_resize(img, (opt.size, opt.size)))
trans_init = [rz_func]
if(opt.crop is not None):
  trans_init = [transforms.CenterCrop(opt.crop),]
  print('Cropping to [%i]'%opt.crop)
else:
  print('Not cropping')

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}


def custom_resize(img, size):
    return TF.resize(img, size, interpolation=rz_dict['bilinear'])

trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset loader
if(type(opt.dir)==str):
  opt.dir = [opt.dir,]

print('Loading [%i] datasets'%len(opt.dir))
data_loaders = []
lengths = []
for dir in opt.dir:
  dataset = datasets.ImageFolder(dir, transform=trans)
  lengths.append(len(dataset))
  data_loaders+=[torch.utils.data.DataLoader(dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=False,
                                          num_workers=opt.workers),]

y_true, y_pred = [], []
Hs, Ws = [], []
len_idxs = [0] + list(np.cumsum(lengths))

with torch.no_grad():
  for idx, data_loader in enumerate(data_loaders):
    for data, label in tqdm(data_loader):

      y_true.extend(label.flatten().tolist())
      if(not opt.size_only):
        if(not opt.use_cpu):
            data = data.to(opt.device)
        y_pred.extend(((model(data).sigmoid()[:,0]>0.5)+0.).flatten().tolist())

    y_true1 = np.array(y_true[len_idxs[idx]:len_idxs[idx+1]])
    y_pred1 = np.array(y_pred[len_idxs[idx]:len_idxs[idx+1]])
    r_acc = accuracy_score(y_true1[y_true1 == 0], y_pred1[y_true1 == 0] > 0.5)
    f_acc = accuracy_score(y_true1[y_true1 == 1], y_pred1[y_true1 == 1] > 0.5)
    acc = accuracy_score(y_true1, y_pred1 > 0.5)
    ap = average_precision_score(y_true1, y_pred1)
    fpr, tpr, thresholds = metrics.roc_curve(y_true1, y_pred1, pos_label=1)
    AUC = metrics.auc(fpr, tpr)

    logger.info('IDX: {} AUC: {:2.2f}, AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(idx, AUC * 100., ap * 100., acc * 100.,
                                                                                       r_acc * 100., f_acc * 100.))
y_true, y_pred = np.array(y_true), np.array(y_pred)

if(not opt.size_only):
  r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
  f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
  acc = accuracy_score(y_true, y_pred > 0.5)
  ap = average_precision_score(y_true, y_pred)
  fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
  AUC = metrics.auc(fpr, tpr)

  logger.info(
      'ALL: AUC: {:2.2f}, AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(AUC * 100., ap * 100.,
                                                                                                 acc * 100.,
                                                                                                 r_acc * 100.,
                                                                                                 f_acc * 100.))


