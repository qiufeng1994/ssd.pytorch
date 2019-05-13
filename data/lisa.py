from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import random
from path import Path
from scipy.misc import imread, imresize
import scipy.io as sio
import cv2


LISA_ROOT = '/Data/LISA_handetect/detectiondata'
# LISA_ROOT = '/Outputs/ssd.pytorch/images_test'
LISA_CLASSES = ('hand')

class LISA(data.Dataset):
    """`LISA_DRIVER_DATASET <http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/#cite1>`_ Dataset.

    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    training_file = 'train'
    test_file = 'test'
    
    def __init__(self, root, train=True, transform=None, target_transform='scale', download=False, vis=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.vis = vis
        self.name = 'LISA'
        if self.train:
            self.train_root = (Path(self.root) / self.training_file / 'pos')
            self.train_samples = self.collect_samples(self.train_root, self.training_file)
        else:
            self.test_root = (Path(self.root) / self.test_file / 'pos') # Jester_pos pos RHD_pos ZJU_pos
            self.test_samples = self.collect_samples(self.test_root, self.test_file)
    
    def collect_samples(self, root, file):
        samples = []
        imgs = sorted((root).glob('*.png'))
        imgs += sorted(root.glob('*.jpg'))
        for img in imgs:
            _img = img.basename().split('.')[0]
            label = (Path(self.root) / file / 'posGt' / _img + '.txt')
            if self.train:
                if not label.exists():
                    continue
                assert label.exists()
            sample = {'img': img, 'label': label}
            samples.append(sample)
        return samples
    
    def load_samples(self, s):
        image = cv2.imread(s['img'])
        try:
            target = list()
            if not s['label'].exists():
                target = [[0,0,0,0,0]]
                print('{}.png has no gt {}'.format(s['img'].namebase, s['label']))
                
            # assert  s['label'].exists()
            else:
                with open(s['label']) as f:
                    label = f.readlines()
                num_objs = len(label) - 1
                for i, obj in enumerate(label[1:]):
                    obj = obj.split(' ')
                    x, y, h, w = map(int, [obj[1], obj[2], obj[3], obj[4]])
                    target.append([x, y, x + h, y + w, 0])
                    # target：# [xmin, ymin, xmax, ymax, label_idx]
        except:
            print('error {}'.format(s))
            
        return [image, target]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, h, w, image_ori = self.pull_item(index)
        return img, target
        
    def __len__(self):
        if self.train:
            return len(self.train_samples)
        else:
            return len(self.test_samples)
        
    def target_scale(self, target, h, w):
        target_trans = []
        scale = np.array([h, w, h, w])
        for i, label in enumerate(target):
            box = list(np.array(label[:4]) / scale)
            box.append(label[4])
            target_trans.append(box)
        return target_trans
        
    def pull_item(self, index):
        if self.train:
            s = self.train_samples[index]
        else:
            s = self.test_samples[index]
    
        image, target = self.load_samples(s)
        # doing this so that it is consistent with all other datasets
        w, h, _ = image.shape
        target = self.target_scale(target, h, w)
    
        # target = Image.fromarray(np.array(image))
        # h, w = img.size[0], img.size[1]
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(image, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
    
        return torch.from_numpy(img).permute(2, 0, 1), target, h, w, image
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



def visual_box(data, output, i):
    import cv2
    import scipy
    img = Image.fromarray(np.array(data).squeeze(), mode='RGB')
    h, w = img.size[0], img.size[1]
    output = np.array(output).squeeze()
    output[::2] = output[::2] * w
    output[1::2] = output[1::2] * h
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    for j in range(0, 8, 2):
        cv2.circle(img, (int(output[j + 1]), int(output[j])), 5, (255, 255, 0), -1)
    # box format (w1, h1, w2, h2, ...)
    cv2.imwrite('/Data/hand_dataset_ox/vis/{:05d}.jpg'.format(i), img)
    print('img saving to \'/Data/hand_dataset_ox/vis/{:05d}.jpg\''.format(i))



def copy_images():
    import os
    import shutil
    root = '/Data/LISA_handetect/detectiondata/test/Jester_pos/'
    dirs = [root + x for x in os.listdir(root) if os.path.isdir(root+x)]
    dirs.sort()
    for dir in dirs:
        idx = Path(dir).stem
        imgs = Path(dir).glob('*.jpg')
        for img in imgs:
            shutil.copyfile(img, root + str(idx) + '_{}.jpg'.format(img.stem))
            print('writing {}_{}.jpg'.format(idx, img.stem))

if __name__ == '__main__':
    # process_hand_mask()
    import shutil
    copy_images()
    # from utils.augmentations import SSDAugmentation
    # # from data import *
    # from data import detection_collate
    #
    # train_set = LISA(LISA_ROOT, train=True, transform=SSDAugmentation(300))
    # train_loader = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=8, shuffle=True,
    #     num_workers=0, collate_fn=detection_collate,
    #     pin_memory=True)
    #
    # for i_step, (input, target) in enumerate(train_loader):
    #     print(input)
    #     print(target)
    #
    # # batch_iterator = iter(train_loader)
    # # for iteration in range(0, 120000):
    # #     images, target = next(batch_iterator)

    # print(1)
    # file = '/Data/LISA_handetect/detectiondata/train/posGt/11_0000530_0_0_0_0.txt'
    # with open(file) as f:
    #     label = f.readlines()
    # print(label)