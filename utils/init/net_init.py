from torch import nn


def net_xavier_uniform_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


if __name__ == '__main__':
    from model.model_zoo import get_model
    net_name = 'ssd_512_mobilenet1.0_coco'
    net = get_model(net_name, pretrained_base=True)
    net_xavier_uniform_init(net)