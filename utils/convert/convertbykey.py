import os
import json
import argparse
import torch


def convert_by_keys(json_file, trained_path, check_name, save_filename):
    with open(json_file, "r") as jsonFile:
        keys_map = json.load(jsonFile)
    state_dict = torch.load(trained_path)
    if check_name is not None:
        state_dict = state_dict[check_name]
    keys = state_dict.keys()
    model_params = {}
    for key in keys:
        if key not in keys_map:
            continue
        model_params[keys_map[key]] = state_dict[key]
        # print(keys_map[key], state_dict[key].shape)
    torch.save(model_params, save_filename)
    print('Finish')


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Convert pytorch model from json keys map')
    parse.add_argument('--name', type=str, default='centernet_dla34_dcn_coco', help='name of the model')
    parse.add_argument('--save-path', type=str, default=os.path.expanduser('~/.torch/models'),
                       help='path to the pytorch models')
    parse.add_argument('--trained-path', type=str,
                       default='/home/ace/cbb/code/dl/CenterNet/models/ctdet_coco_dla_1x.pth',
                       help='trained model')
    parse.add_argument('--check_name', type=str, default='state_dict', help='for pth is not pure state_dict')
    args = parse.parse_args()
    json_file = os.path.join('./key_json', args.name + '.json')
    save_filename = os.path.join(args.save_path, args.name + '.pth')
    convert_by_keys(json_file, args.trained_path, args.check_name, save_filename)
