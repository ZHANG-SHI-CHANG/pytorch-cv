import os
import sys
import argparse
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model.model_zoo import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
    parser.add_argument('--model', type=str, default='ResNet18_v1',
                        help='name of the model to use')
    parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
                        help='path to the saved model parameters')
    parser.add_argument('--saved-params', type=str, default='',
                        help='path to the saved model parameters')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='demo with GPU')
    parser.add_argument('--input-pic', type=str, default=os.path.join(cur_path, '../png/mt_baker.jpg'),
                        help='path to the input picture')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    device = torch.device('cpu')
    if opt.cuda:
        device = torch.device('cuda:0')
    # Load Model
    model_name = opt.model
    pretrained = True if opt.saved_params == '' else False
    net = get_model(model_name, pretrained=pretrained, root=opt.root).to(device)
    net.eval()

    # Load Images
    img = Image.open(opt.input_pic)

    # Transform
    transform_fn = transforms.Compose([
        transforms.Resize(256 if model_name.lower() != 'inceptionv3' else 299),
        transforms.CenterCrop(224 if model_name.lower() != 'inceptionv3' else 299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform_fn(img).to(device)
    with torch.no_grad():
        pred = net(img.unsqueeze(0)).squeeze()

    topK = 5
    _, ind = pred.topk(topK)
    print('The input picture is classified to be')
    for i in range(topK):
        print('\t[%s], with probability %.3f.' %
              (net.classes[ind[i].item()], F.softmax(pred, 0)[ind[i]].item()))
