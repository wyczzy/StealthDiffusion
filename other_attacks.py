import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import network
import timm

import warnings

from efficientnet_pytorch import EfficientNet

warnings.filterwarnings("ignore")

class OURS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, image_input):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        output = self.model(image_input)
        zero_output = torch.zeros_like(output)
        # return torch.cat([torch.tensor([0.]).to(image_input.device), output[0]]).unsqueeze(0).to(image_input.device)
        # return torch.cat([torch.tensor([0.]).to(image_input.device), output[0]]).unsqueeze(0).to(image_input.device)
        return torch.cat((zero_output, output), dim=1).to(image_input.device)
        # return torch.concatenate([torch.tensor([0.]).to(image_input.device), output[0]]).unsqueeze(0).to(image_input.device)
        # return torch.concatenate([output[0], torch.tensor([0.]).to(image_input.device)]).unsqueeze(0).to(image_input.device)
        # return torch.tensor([output[0], 0.]).to(image_input.device)


def model_selection(name):
    dic = {
        "R": "./checkpoints/resnet50.pth",
        "E": "./checkpoints/efficientnet-b0.pth",
        "D": "./checkpoints/deit.pth",
        "S": "./checkpoints/swin-t.pth",
    }
    # dic = {
    #     "R": "/path/to/.pth",
    #     "E": "/path/to/.pth",
    #     "D": "/path/to/.pth",
    #     "S": "/path/to/.pth",
    # }

    if name == "R":
        model_path = dic[name]
        model = network.resnet50(num_classes=1)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        model = OURS(model)
    elif name == "E":
        model_path = dic[name]
        model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=1, image_size=None, )
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        model = OURS(model)
    elif name == "D":
        model_path = dic[name]
        model = timm.create_model(
            'deit_base_patch16_224',
            pretrained=False
        )
        model.head = torch.nn.Linear(in_features=768, out_features=1, bias=True)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict["model"])
        model = OURS(model)
    elif name == "S":
        model_path = dic[name]
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        model.head = torch.nn.Linear(in_features=1024, out_features=1, bias=True)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict["model"])
        model = OURS(model)
    else:
        raise NotImplementedError("No such model!")
    return model.cuda()


