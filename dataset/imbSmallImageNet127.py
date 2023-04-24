from pprint import isreadable
import numpy as np
from PIL import Image
import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'smallimagenet127_x32_train':
            return '/home/weilegexiang/980SSD/ImageNetRoot/ILSVRC2012_img_train_x32/'
        elif dataset == 'smallimagenet127_x32_val':
            return '/home/weilegexiang/980SSD/ImageNetRoot/ILSVRC2012_img_val_x32/'
        elif dataset == 'smallimagenet127_x64_train':
            return '/home/weilegexiang/980SSD/ImageNetRoot/ILSVRC2012_img_train_x64/'
        elif dataset == 'smallimagenet127_x64_val':
            return '/home/weilegexiang/980SSD/ImageNetRoot/ILSVRC2012_img_val_x64/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

# Parameters for data
smallimagenet_mean = (0.48109809, 0.45747185, 0.40785507) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
smallimagenet_std = (0.26040889, 0.2532126, 0.26820634) # equals np.std(train_set.train_data, axis=(0,1,2))/255

# Augmentations.
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(smallimagenet_mean, smallimagenet_std)
    ])

transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(smallimagenet_mean, smallimagenet_std)
])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(smallimagenet_mean, smallimagenet_std)
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

def get_smallimagenet127(args, root, download=False, isrelabel=False):
    train_labeled_idxs, train_unlabeled_idxs = get_split(args)
    img_size = args.img_size
    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=int(img_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(smallimagenet_mean, smallimagenet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomCrop(img_size, padding=int(img_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(smallimagenet_mean, smallimagenet_std)
    ])
    transform_strong.transforms.insert(0, RandAugment(3, 4))
    transform_strong.transforms.append(CutoutDefault(int(img_size / 2)))

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(smallimagenet_mean, smallimagenet_std)
    ])
    train_labeled_idxs, train_unlabeled_idxs = get_split(args)

    train_labeled_dataset = ImbSmallImageNet127(args, Path.db_root_dir('smallimagenet127_x'+str(img_size)+'_train'), indexs=train_labeled_idxs, train=True, transform=transform_train)
    train_unlabeled_dataset = ImbSmallImageNet127(args, Path.db_root_dir('smallimagenet127_x'+str(img_size)+'_train'), indexs=train_unlabeled_idxs, train=True,
                                                transform=TransformTwice(transform_train, transform_strong))
    test_dataset = ImbSmallImageNet127(args, Path.db_root_dir('smallimagenet127_x'+str(img_size)+'_val'), indexs=None, train=False, transform=transform_val)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset,test_dataset, train_labeled_idxs, train_unlabeled_idxs

def get_split(args):
    beta = args.label_ratio
    base_dataset = ImbSmallImageNet127(args, Path.db_root_dir('smallimagenet127_x'+str(args.img_size)+'_train'), indexs=None, train=True, transform=transform_train)
    train_labeled_idxs = []
    for i in range(len(base_dataset.samples)):
        path, target = base_dataset.samples[i]
    train_unlabeled_idxs = np.array(list(set(range(1281167))-set(train_labeled_idxs)))
    print('train_unlabeled_idxs', train_labeled_idxs.shape, train_unlabeled_idxs.shape)
    return train_labeled_idxs, train_unlabeled_idxs


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImbSmallImageNet127(torchvision.datasets.ImageFolder):
    cls_num = 127

    def __init__(self, args, root, indexs=None, rand_number=0, transform=None, target_transform=None, loader=pil_loader, train=True, isrelabel=False):
        super(ImbSmallImageNet127, self).__init__(root=root, transform=transform, target_transform=target_transform, loader=loader)
        self.root = root
        np.random.seed(rand_number)
        self.train = train
        self.targets = []
        self.img_num_list = [0 for i in range(self.cls_num)]
        self.idx = np.array(range(len(self.samples)))

        if not indexs is None:
            new_samples = []
            for idx in indexs:
                new_samples.append(self.samples[idx])
            self.samples = new_samples
            self.idx = indexs
            del new_samples

        for i in range(len(self.samples)):
            path, target = self.samples[i]
            self.targets.append(target)
            self.img_num_list[target] = self.img_num_list[target] + 1

        print('ImbSmallImageNet127_img_num_list: ', self.img_num_list)

        
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.idx[index]
    
    def getAllLabeles(self):
        return self.targets