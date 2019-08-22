import os
import pickle
from skimage.transform import resize

import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms


class MNIST_Dataset():
    def __init__(self, cfg):
        self.dataset_dir = cfg.dataset_dir
        self.prepro_dir = cfg.prepro_dir

        self.num_instances = cfg.num_instances
        self.image_height = cfg.image_height
        self.image_width = cfg.image_width
        self.image_channel_size = cfg.image_channel_size

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.ToPILImage(),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,)),
                                            ])

        self.prepro_train_file = os.path.join(self.prepro_dir, str(self.num_instances), 'mnist_train.pickle')
        self.prepro_test_file = os.path.join(self.prepro_dir, str(self.num_instances), 'mnist_test.pickle')

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.load_dataset()

        if not os.path.exists(self.prepro_dir):
            os.makedirs(self.prepro_dir)

        if not os.path.exists(os.path.join(self.prepro_dir, str(self.num_instances))):
            os.makedirs(os.path.join(self.prepro_dir, str(self.num_instances)))

        if not os.path.exists(self.prepro_train_file):
            self.preprocess_dataset(train=True)
            print('Load train dataset -->', self.prepro_train_file)
            with open(self.prepro_train_file, 'rb') as f:
                self.train_dataset = pickle.load(f)
        else:
            print('Load train dataset -->', self.prepro_train_file)
            with open(self.prepro_train_file, 'rb') as f:
                self.train_dataset = pickle.load(f)

        if not os.path.exists(self.prepro_test_file):
            self.preprocess_dataset(train=False)
            print('Load test dataset -->', self.prepro_test_file)
            with open(self.prepro_test_file, 'rb') as f:
                self.test_dataset = pickle.load(f)
        else:
            print('Load test dataset -->', self.prepro_test_file)
            with open(self.prepro_test_file, 'rb') as f:
                self.test_dataset = pickle.load(f)

    def load_dataset(self):
        self.raw_train_dataset = datasets.MNIST(root=self.dataset_dir,
                                                train=True,
                                                download=True,
                                                transform=None)

        self.raw_test_dataset = datasets.MNIST(root=self.dataset_dir,
                                               train=False,
                                               download=True,
                                               transform=None)

    def preprocess_dataset(self, train=True):
        if train:
            print()
            print('Preprocess train dataset')
            print()
            images = self.raw_train_dataset.data.numpy()
            labels = self.raw_train_dataset.targets.numpy()
        else:
            print()
            print('Preprocess test dataset')
            print()
            images = self.raw_test_dataset.data.numpy()
            labels = self.raw_test_dataset.targets.numpy()

        _dataset = []
        for i, (img, label) in enumerate(zip(images, labels)):
            _dataset.append((img, label))
        if train:
            random.shuffle(_dataset)

        dataset = []
        instance_idx = 0
        for i, (img, label) in enumerate(_dataset):
            img = np.expand_dims(img, axis=2)
            img = resize(img, (self.image_height, self.image_width), anti_aliasing=True)
            img = img.astype(np.float32)
            dataset.append((self.transform(img), label, instance_idx))
            instance_idx += 1
            if self.num_instances <= instance_idx and train:
                break
        self.max_num_instances = len(dataset)
        print('The number of instances: %s' % self.max_num_instances)

        if train:
            with open(self.prepro_train_file, 'wb') as f:
                pickle.dump(dataset, f)
        else:
            with open(self.prepro_test_file, 'wb') as f:
                pickle.dump(dataset, f)


class BatchCollator():
    def __init__(self, image_height, image_width, image_channel_size):
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel_size = image_channel_size

    def __call__(self, batch):
        batch_padded = [b for b in batch]
        return default_collate(batch_padded)
