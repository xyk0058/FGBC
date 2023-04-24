import numpy as np
from PIL import Image

import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

# Parameters for data
stl10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
stl10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

# Augmentations.
# transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(cifar10_mean, cifar10_std)
#     ])

# transform_strong = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(cifar10_mean, cifar10_std)
# ])

# transform_strong.transforms.insert(0, RandAugment(3, 4))
# transform_strong.transforms.append(CutoutDefault(16))


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(stl10_mean, stl10_std),
])

transform_strong = transforms.Compose([
    torchvision.transforms.Resize(32),
    RandAugment(3, 4),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(stl10_mean, stl10_std),
    CutoutDefault(16)
])

transform_val = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.CenterCrop(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(stl10_mean, stl10_std)
])


class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3

def get_stl10(root, l_samples, u_samples, transform_train=transform_train, transform_strong=transform_strong,
                transform_val=transform_val, download=False):
    base_dataset = torchvision.datasets.STL10(root, split='train', download=download)
    print('all labeled data: ', len(base_dataset.labels))
    train_labeled_idxs, train_unlabeled_idxs= train_split(base_dataset.labels, l_samples, u_samples)

    train_labeled_dataset = STL10_labeled(root, train_labeled_idxs, split='train', transform=transform_train, download=download)
    train_unlabeled_dataset = STL10_unlabeled(root, train_unlabeled_idxs, split='unlabeled',
                                                transform=TransformTwice(transform_train, transform_strong), download=download)
    train_unlabeled_dataset2 = STL10_unlabeled2(root, train_unlabeled_idxs, split='train', transform=TransformTwice(transform_train, transform_strong), download=download)
    test_dataset = STL10_labeled(root, split='test', transform=transform_val, download=download)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_dataset.data)} #TestSet: {len(test_dataset.data)}")
    return train_labeled_dataset, train_unlabeled_dataset,test_dataset, train_labeled_idxs, train_unlabeled_idxs, train_unlabeled_dataset2

def train_split(labels, n_labeled_per_class, n_unlabeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    for i in range(10):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs, train_unlabeled_idxs

class STL10_labeled(torchvision.datasets.STL10):

    def __init__(self, root, indexs=None, split='train',
                 transform=None, target_transform=None,
                 download=True):
        super(STL10_labeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        # self.data = [Image.fromarray(img) for img in self.data]
        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index].astype(np.int64)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    def __len__(self):
        return len(self.data)
    
    def getAllLabeles(self):
        return self.labels


class STL10_unlabeled2(torchvision.datasets.STL10):

    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=True):
        super(STL10_unlabeled2, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        # self.data = [Image.fromarray(img) for img in self.data]
        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index].astype(np.int64)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    def __len__(self):
        return len(self.data)


class STL10_unlabeled(torchvision.datasets.STL10):

    def __init__(self, root, indexs, split='unlabeled',
                 transform=None, target_transform=None,
                 download=True):
        super(STL10_unlabeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]
        train_unlabled = STL10_labeled(root, indexs, split='train', transform=transform_train, download=download)
        self.data = self.data + train_unlabled.data
        self.labels = np.concatenate([self.labels, train_unlabled.labels], axis=0)
        # del train_unlabled
        self.train_unlabled = train_unlabled
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index].astype(np.int64)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    def __len__(self):
        return len(self.data)