import os
import torch
from torch import nn

from model.models_zoo.hourglass import Hourglass
from model.ops.corner_pool import top_pool, bottom_pool, left_pool, right_pool
from model.models_zoo.corner_net import CornerPool, pred_module
from model.models_zoo.corner_net import decode

__all__ = ['get_corner_squeeze', 'corner_squeeze_hourglass_coco']


# TODO: move hourglass out
class CornerSqueeze(nn.Module):
    def __init__(self, stacks=2):
        super(CornerSqueeze, self).__init__()
        self.hg = Hourglass(stacks)
        self.tl_modules = nn.ModuleList([CornerPool(256, top_pool, left_pool) for _ in range(stacks)])
        self.br_modules = nn.ModuleList([CornerPool(256, bottom_pool, right_pool) for _ in range(stacks)])

        self.tl_heats = nn.ModuleList([pred_module(80) for _ in range(stacks)])
        self.br_heats = nn.ModuleList([pred_module(80) for _ in range(stacks)])

        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            nn.init.constant_(tl_heat[-1].bias, -2.19)
            nn.init.constant_(br_heat[-1].bias, -2.19)

        self.tl_tags = nn.ModuleList([pred_module(1) for _ in range(stacks)])
        self.br_tags = nn.ModuleList([pred_module(1) for _ in range(stacks)])

        self.tl_offs = nn.ModuleList([pred_module(2) for _ in range(stacks)])
        self.br_offs = nn.ModuleList([pred_module(2) for _ in range(stacks)])

    def _train(self, *xs):
        pass

    def _test(self, *xs, **kwargs):
        image = xs[0]
        cnvs = self.hg(image)

        tl_mod = self.tl_modules[-1](cnvs[-1])
        br_mod = self.br_modules[-1](cnvs[-1])

        tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)
        tl_tag, br_tag = self.tl_tags[-1](tl_mod), self.br_tags[-1](br_mod)
        tl_off, br_off = self.tl_offs[-1](tl_mod), self.br_offs[-1](br_mod)

        outs = [tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off]
        return decode(*outs, **kwargs), tl_heat, br_heat, tl_tag, br_tag

    def forward(self, *xs, **kwargs):
        if self.training:
            return self._train(*xs, **kwargs)
        else:
            return self._test(*xs, **kwargs)


def get_corner_squeeze(name, pretrained=False, root=os.path.expanduser('~/.torch/models'), **kwargs):
    net = CornerSqueeze(**kwargs)
    if pretrained:
        from model.model_store import get_model_file
        full_name = name
        net.load_state_dict(torch.load(get_model_file(full_name, root=root)))
    return net


def corner_squeeze_hourglass_coco(pretrained=False, pretrained_base=False, **kwargs):
    return get_corner_squeeze('corner_squeeze_hourglass_coco', pretrained=pretrained, **kwargs)


if __name__ == '__main__':
    net = CornerSqueeze(2)
    net.eval()
    # for key in net.state_dict().keys():
    #     print(":\"" + key + "\",")
    # print(len(net.state_dict().keys()))

    # import torch
    # a = torch.randn(1, 3, 512, 512)
    # with torch.no_grad():
    #     out = net(a)
