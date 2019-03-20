import shutil
import os

path = os.path.expanduser('~/.torch/datasets/voc')


shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2007'), os.path.join(path, 'VOC2007'))
shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2012'), os.path.join(path, 'VOC2012'))
shutil.rmtree(os.path.join(path, 'VOCdevkit'))

shutil.move(os.path.join(path, 'benchmark_RELEASE'),
                        os.path.join(path, 'VOCaug'))

filenames = ['VOCaug/dataset/train.txt', 'VOCaug/dataset/val.txt']
# generate trainval.txt
with open(os.path.join(path, 'VOCaug/dataset/trainval.txt'), 'w') as outfile:
    for fname in filenames:
        fname = os.path.join(path, fname)
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)