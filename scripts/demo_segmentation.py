import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torchvision import transforms

from model.model_zoo import get_model
from utils.viz.segmentation import get_color_pallete


def parse_args():
    parser = argparse.ArgumentParser(description='Predict ImageNet labels from a given image')
    parser.add_argument('--model', type=str, default='deeplab_resnet101_ade',
                        help='name of the model to use')
    parser.add_argument('--saved-params', type=str, default='',
                        help='path to the saved model parameters')
    parser.add_argument('--input-pic', type=str, default=None,
                        help='path to the input picture')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    # Load Model
    model_name = opt.model
    pretrained = True if opt.saved_params == '' else False
    model = get_model(model_name, pretrained=pretrained, pretrained_base=False)
    model.eval()

    # Load Images
    if opt.input_pic is None:
        img_map = {'voc': 'voc_example.jpg', 'ade': 'ade_example.jpg',
                   'coco': 'voc_example.jpg', 'citys': 'city_example.jpg'}
        opt.input_pic = './png/' + img_map[model_name.split('_')[-1]]
    img = Image.open(opt.input_pic)

    # Transform
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform_fn(img).unsqueeze(0)
    with torch.no_grad():
        output = model.demo(img)

    color_map = {'voc': 'pascal_voc', 'coco': 'pascal_voc',
                 'ade': 'ade20k', 'citys': 'citys'}
    predict = torch.argmax(output, 1).squeeze(0).numpy()
    mask = get_color_pallete(predict, color_map[model_name.split('_')[-1]])
    mask.save('./png/output.png')
    mmask = mpimg.imread('./png/output.png')
    plt.imshow(mmask)
    plt.show()
