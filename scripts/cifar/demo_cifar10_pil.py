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
    parser = argparse.ArgumentParser(description='Predict CIFAR10 classes from a given image')
    parser.add_argument('--model', type=str, default='CIFAR_ResNeXt29_16x64d',
                        help='name of the model to use')
    parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
                        help='pre-trained model root')
    parser.add_argument('--saved-params', type=str, default='',
                        help='path to the saved model parameters')
    parser.add_argument('--cuda', action='store_true', default=False, help='demo with GPU')
    parser.add_argument('--input-pic', type=str, default=os.path.join(cur_path, '../png/cat.jpg'),
                        help='path to the input picture')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    # config
    classes = 10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    args = parse_args()
    device = torch.device('cpu')
    if args.cuda:
        device = torch.device('cuda:0')
    # Load Model
    model_name = args.model
    pretrained = True if args.saved_params == '' else False
    kwargs = {'classes': classes, 'pretrained': pretrained, 'root': args.root, }
    net = get_model(model_name, **kwargs).to(device)
    net.eval()

    # Load Images
    img = Image.open(args.input_pic)

    # Transform
    transform_fn = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    img = transform_fn(img).to(device)
    with torch.no_grad():
        pred = net(img.unsqueeze(0)).squeeze()

    ind = pred.argmax()
    print('The input picture is classified to be [%s], with probability %.3f.' %
          (class_names[ind.item()], F.softmax(pred, 0)[ind].item()))
