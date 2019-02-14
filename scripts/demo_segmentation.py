import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torchvision import transforms

from model.model_zoo import get_model
from utils.viz.segmentation import get_color_pallete

parser = argparse.ArgumentParser(description='Predict ImageNet labels from a given image')
parser.add_argument('--model', type=str, default='fcn_resnet101_ade',
                    help='name of the model to use')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')
parser.add_argument('--input-pic', type=str, default='./png/example.jpg',
                    help='path to the input picture')
opt = parser.parse_args()

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
model = get_model(model_name, pretrained=pretrained, pretrained_base=False)
model.eval()

# Load Images
img = Image.open(opt.input_pic)

# Transform
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = transform_fn(img).unsqueeze(0)
with torch.no_grad():
    output = model.demo(img)

predict = torch.argmax(output, 1).squeeze(0).numpy()
mask = get_color_pallete(predict, 'pascal_voc')
mask.save('./png/output.png')
mmask = mpimg.imread('./png/output.png')
plt.imshow(mmask)
plt.show()
