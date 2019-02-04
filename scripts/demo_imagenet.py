import argparse
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from model.model_zoo import get_model

parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--model', type=str, default='darknet53',
                    help='name of the model to use')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')
parser.add_argument('--input-pic', type=str, default='./png/mt_baker.jpg',
                    help='path to the input picture')
opt = parser.parse_args()

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
net = get_model(model_name, pretrained=pretrained)
net.eval()

# Load Images
img = Image.open(opt.input_pic)

# Transform
transform_fn = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = transform_fn(img).unsqueeze(0)
with torch.no_grad():
    pred = net(img).squeeze()

topK = 5
_, ind = pred.topk(topK)
print('The input picture is classified to be')
for i in range(topK):
    print('\t[%s], with probability %.3f.' %
          (net.classes[ind[i].item()], F.softmax(pred, 0)[ind[i]].item()))
