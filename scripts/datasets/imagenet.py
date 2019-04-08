import os
import tarfile
import pickle
import gzip
import subprocess

from utils.filesystem import makedirs
from utils.download import download


def build_rec_process(img_dir, train=False, num_thread=1):
    rec_dir = os.path.abspath(os.path.join(img_dir, '../rec'))
    makedirs(rec_dir)
    prefix = 'train' if train else 'val'
    print('Building ImageRecord file for ' + prefix + ' ...')
    to_path = rec_dir

    # download lst file and im2rec script
    script_path = os.path.join(rec_dir, 'im2rec.py')
    script_url = 'https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py'
    download(script_url, script_path)

    lst_path = os.path.join(rec_dir, prefix + '.lst')
    lst_url = 'http://data.mxnet.io/models/imagenet/resnet/' + prefix + '.lst'
    download(lst_url, lst_path)

    # execution
    import sys
    cmd = [
        sys.executable,
        script_path,
        rec_dir,
        img_dir,
        '--recursive',
        '--pass-through',
        '--pack-label',
        '--num-thread',
        str(num_thread)
    ]
    subprocess.call(cmd)
    os.remove(script_path)
    os.remove(lst_path)
    print('ImageRecord file for ' + prefix + ' has been built!')


def extract_val(tar_fname, target_dir, with_rec=False, num_thread=1):
    os.makedirs(target_dir)
    print('Extracting ' + tar_fname)
    with tarfile.open(tar_fname) as tar:
        tar.extractall(target_dir)
    # build rec file before images are moved into subfolders
    if with_rec:
        build_rec_process(target_dir, False, num_thread)
    # move images to proper subfolders
    val_maps_file = os.path.join(os.path.dirname(__file__), 'imagenet_val_maps.pklz')
    with gzip.open(val_maps_file, 'rb') as f:
        dirs, mappings = pickle.load(f)
    for d in dirs:
        os.makedirs(os.path.join(target_dir, d))
    for m in mappings:
        os.rename(os.path.join(target_dir, m[0]), os.path.join(target_dir, m[1], m[0]))


def main(download_dir, target_dir):
    _VAL_TAR = 'ILSVRC2012_img_val.tar'
    val_tar_fname = os.path.join(download_dir, _VAL_TAR)
    target_dir = os.path.expanduser(target_dir)
    extract_val(val_tar_fname, target_dir)


if __name__ == '__main__':
    download_dir = os.path.expanduser('~/cbb/data/ILSVRC2012')
    target_dir = os.path.expanduser('~/cbb/data/imagenet/val')
    main(download_dir, target_dir)