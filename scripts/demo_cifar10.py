import argparse
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from model.model_zoo import get_model

parser = argparse.ArgumentParser(description='Predict CIFAR10 classes from a given image')
parser.add_argument('--model', type=str, default='cifar_resnet110_v1',
                    help='name of the model to use')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')
parser.add_argument('--input-pic', type=str, default='./png/mt_baker.jpg',
                    help='path to the input picture')
opt = parser.parse_args()

classes = 10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
kwargs = {'classes': classes, 'pretrained': pretrained}
net = get_model(model_name, **kwargs)
net.eval()

# Load Images
img = Image.open(opt.input_pic)

# Transform
transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

img = transform_fn(img)
with torch.no_grad():
    pred = net(img.unsqueeze(0))

ind = pred.argmax()
print('The input picture is classified to be [%s], with probability %.3f.' %
      (class_names[ind.item()], F.softmax(pred, 0)[ind].item()))

