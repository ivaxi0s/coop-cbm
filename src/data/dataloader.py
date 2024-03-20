import os

import numpy as np
import pandas as pd
from numpy import genfromtxt

import torch
import torchvision as tv
import torchvision.transforms.functional as TF

from PIL import Image
from torchvision import datasets
from torchvision.utils import save_image

OUT_DIR = 'results'


# create an output folder
os.makedirs(OUT_DIR, exist_ok=True)

class OODBird(datasets.VisionDataset):
  def __init__(self, args, root='./data', env='train', transform=None, target_transform=None):
    super(OODBird, self).__init__(root, transform=transform, target_transform=target_transform)
    self.envs = env
    self.prepare_ood_bird()
    self.data_label_tuples = torch.load(os.path.join(self.root, self.envs) + '.pt')
  
  def __getitem__(self, index):
    img, target, a = self.data_label_tuples[index]
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img, target, a
  
  def __len__(self):
    return len(self.data_label_tuples)
  
  def prepare_ood_bird(self):
    if os.path.exists(os.path.join(self.root, 'train.pt')) \
      and os.path.exists(os.path.join(self.root,'out_test.pt')) \
      and os.path.exists(os.path.join(self.root, 'in_test.pt')):
      
      print('OOD Bird dataset already exists')
      # pdb.set_trace()
      return
    
    print('Preparing OOD Bird')
    train_bird = datasets.ImageFolder(args.root + "/train")
    test_in_bird = datasets.ImageFolder(args.root + "/test_in")
    test_out_bird = datasets.ImageFolder(args.root + "/test_out")
    train_set = []
    in_test_set = []
    out_test_set = []
    
    C_A = np.zeros((200,312))
    class_attributes_file = os.path.join('/lustre04/scratch/ivsh/datasets/CUB/CUB_200_2011', 'attributes/class_attribute_labels_continuous.txt')
    class_attr_rf = open(class_attributes_file,'r')
    i = 0
    for line in class_attr_rf.readlines():
      strs = line.strip().split(' ')
      for j in range(len(strs)):
        strs[j] = float(strs[j])
        C_A[i][j] = 0.0 if strs[j] < 50.0 else 1.0
      i+=1
    class_attr_rf.close()
    
    for idx, (im, label) in enumerate(train_bird):
      print(label)
      if label < 200:
        if idx % 1000 == 0:
          print(f'Converting {idx}/{len(train_bird)}')
        im_array = np.array(im)

        train_set.append((im, label, C_A[label]))

    for idx, (im, label) in enumerate(test_in_bird):
      if label < 200:
        if idx % 1000 == 0:
          print(f'Converting {idx}/{len(test_in_bird)}')
        im_array = np.array(im)

        in_test_set.append((im, label, C_A[label]))

    for idx, (im, label) in enumerate(test_out_bird):
      if label < 200:
        if idx % 1000 == 0:
          print(f'Converting {idx}/{len(test_out_bird)}')
        im_array = np.array(im)
        
        out_test_set.append((im, label, C_A[label]))
  
    torch.save(train_set, os.path.join(self.root, 'train.pt'))
    torch.save(in_test_set, os.path.join(self.root, 'in_test.pt'))
    torch.save(out_test_set, os.path.join(self.root, 'out_test.pt'))

class DDI(datasets.VisionDataset):
  def __init__(self, root, transform=None):
    self.root = root
    self.df = pd.read_csv(os.path.join(self.root,'ddi_metadata.csv'))
    self.attr_df = pd.read_csv(os.path.join(self.root,'ddi_attributes.csv'))
    self.attr_txt = genfromtxt(os.path.join(self.root, 'attributes.txt'))

    self.transform = transform
  def __len__(self):
    return(len(self.attr_df))

  def __getitem__(self, index):
    # print(index)
    curr_df = self.attr_df.iloc[index]
    img_name = curr_df['ImageID']
    is_malignant = int(self.df.loc[self.df['DDI_file'] == curr_df['ImageID']]['malignant'].item()) #y

    X = Image.open(os.path.join(self.root, img_name))
    y = torch.tensor(is_malignant)
    a = torch.tensor(self.attr_txt[index])
    if self.transform:
      X = self.transform(X)
      if X.shape[0] == 4:
        X = X[:3,:,:]
    return X,y

class DatasetBirds(tv.datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    """
    def __init__(self,
                 root,
                 args,
                 transform=None,
                 target_transform=None,
                 loader=tv.datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):
        img_root = os.path.join(root, 'images')

        super(DatasetBirds, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        self.args = args
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images

        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)
        # pdb.set_trace()
        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use
        global C_A
        C_A = np.zeros((200,312))
        class_attributes_file = os.path.join(root, 'attributes/class_attribute_labels_continuous.txt') 
        class_attr_rf = open(class_attributes_file,'r')
        i = 0
        for line in class_attr_rf.readlines():
            strs = line.strip().split(' ')
            for j in range(len(strs)):
                strs[j] = float(strs[j])
                C_A[i][j] = 0.0 if strs[j] < 50.0 else 1.0
            i+=1
        class_attr_rf.close()
        # pdb.set_trace()
        if args.repeat_concepts:
          concepts_repeated = int(args.rep*args.n_attributes)
          intm_con = C_A[:,:concepts_repeated]
          C_A = np.hstack((C_A, intm_con))

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(DatasetBirds, self).__getitem__(index)

        if self.bboxes is not None:
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)
        attribute = C_A[target]
        if self.args.corruption_name:
          sample = np.array(sample)
          sample = sample.transpose(1, 2, 0)
          sample = (sample* 255).astype(np.uint8)
          sample = corrupt(sample, corruption_name=self.args.corruption_name, severity=1)
          pil_image = Image.fromarray(sample)
          pil_image.save("output_image.png")  
          sample = sample.transpose(2, 0, 1)
          sample = torch.from_numpy(sample)
        return sample, target, attribute


def pad(img, fill=0, size_max=500):
    """
    Pads images to the specified size (height x width). 
    Fills up the padded area with value(s) passed to the `fill` parameter. 
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)
    
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

class TIL(tv.datasets.ImageFolder):
    def __init__(self,
                 root,
                 args,
                 transform=None,
                 target_transform=None,
                 loader=tv.datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):
        img_root = os.path.join(root, 'images')

        super(TIL, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        self.args = args
        
        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) != self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images

        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)
        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs  = self.samples = imgs_to_use
        self.targets = targets_to_use
        global C_A
        C_A = np.zeros((4520,185))
        class_attributes_file = os.path.join(root, 'attributes_medical.txt')
        class_attr_rf = open(class_attributes_file,'r')
        i = 0
        for line in class_attr_rf.readlines():
            strs = line.strip().split(' ')
            for j in range(len(strs)):
                strs[j] = float(strs[j])
                C_A[i][j] = strs[j]
            i+=1
        class_attr_rf.close()
        if args.repeat_concepts:
          concepts_repeated = int(args.rep*args.n_attributes)
          intm_con = C_A[:,:concepts_repeated]
          C_A = np.hstack((C_A, intm_con))

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(TIL, self).__getitem__(index)

        if self.bboxes is not None:
            # squeeze coordinates of the bounding box to range [0, 1]
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)
        attribute = C_A[index]
        return sample, target

class DatasetAnimals(tv.datasets.ImageFolder):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=tv.datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):
        img_root = os.path.join(root, 'images')

        super(DatasetAnimals, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) != self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images

        path_to_index = os.path.join(root, 'all.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)
        # pdb.set_trace()
        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use
        global C_A
        C_A = np.zeros((200,312))
        class_attributes_file = os.path.join(root, 'predicate-matrix-binary.txt') 
        class_attr_rf = open(class_attributes_file,'r')
        i = 0
        for line in class_attr_rf.readlines():
            strs = line.strip().split(' ')
            for j in range(len(strs)):
                strs[j] = float(strs[j])
                C_A[i][j] = strs[j]
                # C_A[i][j] = strs[j]
            i+=1
        class_attr_rf.close()
        # pdb.set_trace()

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(DatasetAnimals, self).__getitem__(index)

        if self.bboxes is not None:
            # squeeze coordinates of the bounding box to range [0, 1]
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)
        attribute = C_A[target]
        # attribute = C_A[index]
        # pdb.set_trace()
        return sample, target, attribute

class Nec(tv.datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=tv.datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):
        img_root = os.path.join(root, 'images')

        super(Nec, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        
        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) != self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images

        path_to_index = os.path.join(root, 'nec.txt')
        filenames_to_use = set()
        label = []
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                # pdb.set_trace()
                l, idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)
                    label.append(l)

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))
        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use
        global C_A
        C_A = np.zeros((4520,183))
        class_attributes_file = os.path.join(root, 'meta.txt')
        class_attr_rf = open(class_attributes_file,'r')
        i = 0
        for line in class_attr_rf.readlines():
            strs = line.strip().split(' ')
            for j in range(len(strs)):
                strs[j] = float(strs[j])
                C_A[i][j] = strs[j]
            i+=1
        class_attr_rf.close()
        global nec
        nec = np.zeros((4520,185))
        class_attributes_file = os.path.join(root, 'attributes_medical.txt')
        class_attr_rf = open(class_attributes_file,'r')
        i = 0
        for line in class_attr_rf.readlines():
            strs = line.strip().split(' ')
            for j in range(len(strs)):
                strs[j] = float(strs[j])
                # C_A[i][j] = 0.0 if strs[j] < 50.0 else 1.0
                nec[i][j] = strs[j]
            i+=1
        class_attr_rf.close()

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(Nec, self).__getitem__(index)

        if self.bboxes is not None:
            # squeeze coordinates of the bounding box to range [0, 1]
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)
        nec_target = nec[index][96]
        attribute = C_A[index]
        return sample, nec_target, attribute

def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr

class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for OOD. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf for out of distribution dataset.
  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train', 'in_test' or 'out_test.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data', env='train', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)
    
    self.envs = env
    self.prepare_colored_mnist()
    if env in ['train', 'out_test', 'in_test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train, in_test, out_test')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target, a = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target, a

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if not os.path.isdir(colored_mnist_dir):
        os.mkdir(colored_mnist_dir)
    # pdb.set_trace()
    if os.path.exists(os.path.join(colored_mnist_dir, 'train.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'out_test.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'in_test.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
    test_mnist = datasets.mnist.MNIST(self.root, train=False, download=True)
    train_set = []
    in_test_set = []
    out_test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
        im_array = np.array(im)
        
      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1
      attribute = torch.zeros(10)
      attribute[label] = 1
      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
        # 20% in the first training environment	
      if np.random.uniform() < 0.2:	
        color_red = not color_red
      colored_arr = color_grayscale_arr(im_array, red=color_red)
      train_set.append((Image.fromarray(colored_arr), binary_label, attribute))
    
    for idx, (im, label) in enumerate(test_mnist):
      test_im_array = np.array(im)
      
      binary_label = 0 if label < 5 else 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0
      attribute = torch.zeros(10)
      attribute[label] = 1
      # 90% in the out_test environment
      color_red_out = color_red	
      if np.random.uniform() < 0.9:	
        color_red_out = not color_red
        colored_arr = color_grayscale_arr(test_im_array, red=color_red_out)
      out_test_set.append((Image.fromarray(colored_arr), binary_label, attribute))

      # 20% in the in_test environment	
      color_red_in = color_red		
      if np.random.uniform() < 0.2:	
        color_red_in = not color_red
      colored_arr = color_grayscale_arr(test_im_array, red=color_red_in)
      in_test_set.append((Image.fromarray(colored_arr), binary_label, attribute))

    torch.save(train_set, os.path.join(colored_mnist_dir, 'train.pt'))
    torch.save(in_test_set, os.path.join(colored_mnist_dir, 'in_test.pt'))
    torch.save(out_test_set, os.path.join(colored_mnist_dir, 'out_test.pt'))


class CelebA(datasets.VisionDataset):
  def __init__(self, root='./data', env='train', transform=None, target_transform=None):
    super(CelebA, self).__init__(root, transform=transform, target_transform=target_transform)
    # pdb.set_trace()
    
    self.envs = env
    self.prepare_celeba()
    self.data_label_tuples = torch.load(os.path.join(self.root, 'img_align_celeba', env) + '.pt')
  
  def __getitem__(self, index):
    img, target, attribute = self.data_label_tuples[index]
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img, target, attribute
  
  def __len__(self):
    return len(self.data_label_tuples)
  
  def prepare_celeba(self):
    celeb_root = os.path.join(self.root, 'img_align_celeba')
    # pdb.set_trace()
    if os.path.exists(os.path.join(self.root, 'img_align_celeba', 'train.pt')) \
      and os.path.exists(os.path.join(self.root, 'img_align_celeba', 'out_test.pt')) \
      and os.path.exists(os.path.join(self.root, 'img_align_celeba', 'in_test.pt')):
      
      print('CelebA already exists')
      return
    
    print('Preparing CelebA')
    train_set = []
    in_test_set = []
    out_test_set = []

    train_blonde_boy = np.loadtxt('celeb_data/train_blonde_boy.txt', dtype='str')
    train_blonde_boy = [os.path.join(celeb_root, x) for x in train_blonde_boy]
    train_blonde_boy = np.c_[train_blonde_boy, np.zeros(len(train_blonde_boy))]
    
    train_black_boy = np.loadtxt('celeb_data/train_black_boy.txt', dtype = 'str')
    train_black_boy = [os.path.join(celeb_root, x) for x in train_black_boy]
    train_black_boy = np.c_[train_black_boy, np.zeros(len(train_black_boy))]

    train_blonde_girl = np.loadtxt('celeb_data/train_blonde_girl.txt', dtype = 'str')
    train_blonde_girl = [os.path.join(celeb_root, x) for x in train_blonde_girl]
    train_blonde_girl = np.c_[train_blonde_girl, np.ones(len(train_blonde_girl))]

    train_black_girl = np.loadtxt('celeb_data/train_black_girl.txt', dtype = 'str')
    train_black_girl = [os.path.join(celeb_root, x) for x in train_black_girl]
    train_black_girl = np.c_[train_black_girl, np.ones(len(train_black_girl))]
    train = np.vstack((train_blonde_boy , train_black_boy , train_blonde_girl , train_black_girl))


    intest_blonde_boy = np.loadtxt('celeb_data/intest_blonde_boy.txt', dtype = 'str')
    intest_blonde_boy = [os.path.join(celeb_root, x) for x in intest_blonde_boy]
    intest_blonde_boy = np.c_[intest_blonde_boy, np.zeros(len(intest_blonde_boy))]

    intest_black_boy = np.loadtxt('celeb_data/intest_black_boy.txt', dtype = 'str')
    intest_black_boy = [os.path.join(celeb_root, x) for x in intest_black_boy]
    intest_black_boy = np.c_[intest_black_boy, np.zeros(len(intest_black_boy))]

    intest_blonde_girl = np.loadtxt('celeb_data/intest_blonde_girl.txt', dtype = 'str')
    intest_blonde_girl = [os.path.join(celeb_root, x) for x in intest_blonde_girl]
    intest_blonde_girl = np.c_[intest_blonde_girl, np.ones(len(intest_blonde_girl))]

    intest_black_girl = np.loadtxt('celeb_data/intest_black_girl.txt', dtype = 'str')
    intest_black_girl = [os.path.join(celeb_root, x) for x in intest_black_girl]
    intest_black_girl = np.c_[intest_black_girl, np.ones(len(intest_black_girl))]
    intest = np.vstack((intest_blonde_boy , intest_black_boy , intest_blonde_girl , intest_black_girl))


    outtest_blonde_boy= np.loadtxt('celeb_data/outtest_blonde_boy.txt', dtype = 'str')
    outtest_blonde_boy = [os.path.join(celeb_root, x) for x in outtest_blonde_boy]
    outtest_blonde_boy = np.c_[outtest_blonde_boy, np.zeros(len(outtest_blonde_boy))]

    outtest_black_boy = np.loadtxt('celeb_data/outtest_black_boy.txt', dtype = 'str')
    outtest_black_boy = [os.path.join(celeb_root, x) for x in outtest_black_boy]
    outtest_black_boy = np.c_[outtest_black_boy, np.zeros(len(outtest_black_boy))]

    outtest_blonde_girl = np.loadtxt('celeb_data/outtest_blonde_girl.txt', dtype = 'str')
    outtest_blonde_girl = [os.path.join(celeb_root, x) for x in outtest_blonde_girl]
    outtest_blonde_girl = np.c_[outtest_blonde_girl, np.ones(len(outtest_blonde_girl))]

    outtest_black_girl = np.loadtxt('celeb_data/outtest_black_girl.txt', dtype = 'str')
    outtest_black_girl = [os.path.join(celeb_root, x) for x in outtest_black_girl]
    outtest_black_girl = np.c_[outtest_black_girl, np.ones(len(outtest_black_girl))]
    outtest = np.vstack((outtest_blonde_boy , outtest_black_boy , outtest_blonde_girl , outtest_black_girl))
    all_attributes = np.loadtxt("celeb_data/no_hair_attributes.txt", dtype = int)
    print(len(train))
    for i in range(len(train)):
      # pdb.set_trace()
      idx = int(train[i][0][-10:-4])
      attr = all_attributes[idx-1]
      print(i)
      train_set.append((Image.open(train[i][0]), int(float(train[i][1])), attr))
    
    print(len(intest))
    for i in range(len(intest)):
      idx = int(intest[i][0][-10:-4])
      attr = all_attributes[idx-1]
      print(i)
      in_test_set.append((Image.open(intest[i][0]), int(float(intest[i][1])), attr))

    print(len(outtest))
    for i in range(len(outtest)):
      idx = int(outtest[i][0][-10:-4])
      attr = all_attributes[idx-1]
      print(i)
      out_test_set.append((Image.open(outtest[i][0]), int(float(outtest[i][1])), attr))

    torch.save(train_set, os.path.join(celeb_root, 'train.pt'))
    torch.save(in_test_set, os.path.join(celeb_root, 'in_test.pt'))
    torch.save(out_test_set, os.path.join(celeb_root, 'out_test.pt'))
