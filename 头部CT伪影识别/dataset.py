from pylab import *
import pydicom
import cv2
import os
from sklearn.model_selection import train_test_split
import sklearn.utils
from collections import Counter
import torch.utils.data as data
import torchvision.transforms as transforms
from imblearn.over_sampling import SMOTE
import torch.nn.functional as F

random_seed = 321  # 随机种子
ratio = 0.1  # 验证集、测试集比例


# 图像预处理方法
def windows_pro(img, min_bound=0, max_bound=85):
    img[img > max_bound] = max_bound
    img[img < min_bound] = min_bound
    img = img - min_bound
    img = normalize(img)
    return img


def equalize_hist(img):
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    return img


def img_resize(img, size=224):
    img = cv2.resize(img, (size, size))
    return img


def normalize(img):
    img = img.astype(np.float32)
    np.seterr(divide='ignore', invalid='ignore')
    img = (img - img.min()) / (img.max() - img.min())
    img = img * 255
    img = img.astype(np.uint8)
    return img


def extend_channels(img):
    img_channels = np.zeros([img.shape[0], img.shape[1], 3])
    img_channels[:, :, 0] = img
    img_channels[:, :, 1] = img
    img_channels[:, :, 2] = img
    return img_channels


def data_preprocess_base(img, size):
    img = img_resize(img, size)
    img = normalize(img)
    img = extend_channels(img)
    img = img.astype(np.uint8)
    return img


def data_preprocess_enhanced(img, size):
    img = equalize_hist(img)
    img = img_resize(img, size)
    img = normalize(img)
    img = extend_channels(img)
    img = img.astype(np.uint8)
    return img


# 读取数据并划分数据集
def data_load(path, test_path, size, is_train, is_sampling='no_sampler'):
    dicomlist = []
    labels = []
    train_img = []
    train_label = []
    val_img = []
    val_label = []

    f = open(path, "r+") if test_path == '' else open(test_path, "r+")
    for line in f.readlines():
        img_path = line.strip().split(',')[0]
        dicomlist.append(img_path)
        label = line.strip().split(',')[1]
        label = '0' if label == 'good' else '1'
        labels.append(label)
    labels = np.array(labels)
    images = array([data_preprocess_enhanced(pydicom.read_file(dcm).pixel_array, size) for dcm in dicomlist])
    f.close()

    if is_train or test_path == '':
        images, labels = sklearn.utils.shuffle(images, labels, random_state=random_seed)
        train_val_img, test_img, train_val_label, test_label = train_test_split(images, labels, test_size=ratio,
                                                                                stratify=labels,
                                                                                random_state=random_seed)
        train_img, val_img, train_label, val_label = train_test_split(train_val_img, train_val_label, test_size=ratio,
                                                                      stratify=train_val_label,
                                                                      random_state=random_seed)
        if is_train:
            if is_sampling == 'over_sampler':
                train_img, train_label = over_sampling(train_img, train_label)
            elif is_sampling == 'down_sampler':
                train_img, train_label = under_sampling(train_img, train_label)
            elif is_sampling == 'smote':
                train_img, train_label = over_sampling_smote(train_img, train_label)
            print('Sampling mode:%s, train_num:%s,label:%s' % (
                is_sampling, train_img.shape, sorted(Counter(train_label).items())))
            print('Dataset: %s, labels=%s' % (images.shape, sorted(Counter(labels).items())))
            print('Training set: %s, labels=%s' % (train_img.shape, sorted(Counter(train_label).items())))
            print('Val set: %s, labels=%s' % (val_img.shape, sorted(Counter(val_label).items())))
            print('Test set: %s, labels=%s' % (test_img.shape, sorted(Counter(test_label).items())))

    else:
        test_img = images
        test_label = labels
        print('Test set: %s, labels=%s' % (test_img.shape, sorted(Counter(test_label).items())))
    return train_img, train_label, val_img, val_label, test_img, test_label


class TrainDataset(data.Dataset):
    def __init__(self, train_img, train_label, train_data_transform=None, minority_class=1):
        super(TrainDataset, self).__init__()
        self.train_img = train_img
        self.train_label = train_label
        self.train_data_transform = train_data_transform
        self.minority_class = minority_class

    def __getitem__(self, index):
        img = self.train_img[index]
        target = int(self.train_label[index])
        if self.train_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))
            if target == self.minority_class:
                augmented_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(30),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    transforms.ToTensor()
                ])
                img = augmented_transforms(img)
            else:
                img = self.train_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.train_img)


class ValDataset(data.Dataset):
    def __init__(self, val_img, val_label, val_data_transform):
        super(ValDataset, self).__init__()
        self.val_img = val_img
        self.val_label = val_label
        self.val_data_transform = val_data_transform

    def __getitem__(self, index):
        img = self.val_img[index]
        target = int(self.val_label[index])
        if self.val_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))
            img = self.val_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.val_img)


class TestDataset(data.Dataset):
    def __init__(self, test_img, test_label, test_data_transform):
        super(TestDataset, self).__init__()
        self.test_img = test_img
        self.test_label = test_label
        self.test_data_transform = test_data_transform

    def __getitem__(self, index):
        img = self.test_img[index]
        target = int(self.test_label[index])
        if self.test_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))
            img = self.test_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.test_img)


def get_dataset(path, test_path, size, batch_size, is_train, is_sampling=False):
    train_img, train_label, val_img, val_label, test_img, test_label = data_load(path, test_path, size, is_train,
                                                                                 is_sampling)
    train_loader = []
    val_loader = []

    if is_train:
        train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 180)),
            transforms.ToTensor()])

        train_set = TrainDataset(train_img, train_label, train_data_transform)
        train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

        val_data_transform = transforms.Compose([
            transforms.ToTensor()])
        val_set = ValDataset(val_img, val_label, val_data_transform)
        val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    test_data_transform = transforms.Compose([
        transforms.ToTensor()])
    test_set = TestDataset(test_img, test_label, test_data_transform)
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def under_sampling(train_img, train_label):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=random_seed, replacement=False)
    nsamples, nx, ny, nz = train_img.shape
    train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
    X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
    return X_resampled, y_resampled


def over_sampling(train_img, train_label):
    from imblearn.over_sampling import RandomOverSampler
    rus = RandomOverSampler(random_state=random_seed)
    nsamples, nx, ny, nz = train_img.shape
    train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
    X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
    return X_resampled, y_resampled


def over_sampling_smote(train_img, train_label):
    smote = SMOTE(random_state=random_seed, k_neighbors=3)
    nsamples, nx, ny, nz = train_img.shape
    train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
    X_resampled, y_resampled = smote.fit_resample(train_img_flatten, train_label)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
    return X_resampled, y_resampled
