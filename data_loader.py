from PIL import Image
import os
import os.path
import errno
import numpy as np
import codecs
import torch
import torch.utils.data as data

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return parsed#torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
    
class Custom_Mnist_Dataset(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
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
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    eps = .001
    noise = .1

    def __init__(self, root, mode='baseline', train=True, transform=None, target_transform=None, refresh=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.training_file = 'training_%s.pt' % mode
        self.test_file = 'test_%s.pt' % mode
        assert mode in ['baseline', 'correlated', 'missing', 'context-no-info', 'img-no-info', 'multiple']

        if self._check_exists() and refresh:
            os.remove(os.path.join(self.root, self.processed_folder, self.training_file))
            os.remove(os.path.join(self.root, self.processed_folder, self.test_file))
            
        if not self._check_exists():
            self.download(mode)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.imgs, self.huy = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.imgs, self.huy = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.imgs[index], self.huy[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self, mode):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if not os.path.exists(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')):
            for url in self.urls:
                print('Downloading ' + url)
                data = urllib.request.urlopen(url)
                filename = url.rpartition('/')[2]
                file_path = os.path.join(self.root, self.raw_folder, filename)
                with open(file_path, 'wb') as f:
                    f.write(data.read())
                with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                        gzip.GzipFile(file_path) as zip_f:
                    out_f.write(zip_f.read())
                os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        
        def get_vars(path):
            h = read_label_file(path)
            if mode == 'correlated':
                u = np.random.binomial(h,.7)
            elif mode == 'missing':
                u = np.random.binomial(h,.7)
                hide = np.random.binomial(1,(h-u+1)/(h+3))
            else:
                u = np.random.randint(0,10,h.shape)

            if mode == 'context-no-info':
                y = np.round(np.clip(h*2 + np.random.normal(0,self.noise,h.shape), 0,18)/3)
            elif mode == 'img-no-info':
                y = np.round(np.clip(u*2 + np.random.normal(0,self.noise,h.shape), 0,18)/3)
            elif mode == 'multiple':
                v = np.random.binomial(6,(h+1)/11)
                w = np.random.binomial(6,(10-u)/11)
                y = np.round(np.clip(h + u - v + w + np.random.normal(0,self.noise,h.shape), 0,18)/3)
                #v = np.random.chisquare(4,h.shape) #model as gaussian
                #v -= v.mean()
                #w = np.random.binomial(5,(h+1)/11) #model as categorical
                #y = np.round(np.clip(h + u - v + w*(w==3+w==4) + np.random.normal(0,self.noise,h.shape), 0,18)/3)
                v = np.clip(v/6, self.eps, 1-self.eps)
                w = np.clip(w/6, self.eps, 1-self.eps)
            else:
                y = np.round(np.clip(h + u + np.random.normal(0,self.noise,h.shape), 0,18)/3)
            
            if mode == 'missing':
                u = np.clip(u/9, self.eps, 1-self.eps)
                u[hide == 1] = -1
            else:
                u = np.clip(u/9, self.eps, 1-self.eps)
            y = np.clip(y/6, self.eps, 1-self.eps)
            if mode == 'multiple':
                huvwy = np.stack([h, u, v, w, y], 1)
                return torch.from_numpy(huvwy)
            else:
                huy = np.stack([h, u, y], 1)
                return torch.from_numpy(huy)
                
                
        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            get_vars(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            get_vars(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

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
    
    
class Gasros_Dataset(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    eps = .001

    def __init__(self, root, mode='baseline', train=True, transform=None, target_transform=None, refresh=False):
        self.noise = .5
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.training_file = 'training_%s.pt' % mode
        self.test_file = 'test_%s.pt' % mode
        assert mode in ['baseline', 'correlated', 'missing', 'context-no-info', 'img-no-info', 'multiple']

        if self._check_exists() and refresh:
            os.remove(os.path.join(self.root, self.processed_folder, self.training_file))
            os.remove(os.path.join(self.root, self.processed_folder, self.test_file))
            
        if not self._check_exists():
            self.download(mode)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.imgs, self.huy = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.imgs, self.huy = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.imgs[index], self.huy[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self, mode):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        
        def get_vars(path):
            h = read_label_file(path)
            if mode == 'correlated':
                u = np.random.binomial(h,.7)
            elif mode == 'missing':
                u = np.random.binomial(h,.7)
                hide = np.random.binomial(1,(h-u+1)/(h+3))
                u[hide == 1] = -1
            else:
                u = np.random.randint(0,10,h.shape)

            if mode == 'context-no-info':
                y = np.round(np.clip(h*2 + np.random.normal(0,self.noise,h.shape), 0,18)/3)
            elif mode == 'img-no-info':
                y = np.round(np.clip(u*2 + np.random.normal(0,self.noise,h.shape), 0,18)/3)
            elif mode == 'multiple':
                v = np.random.randn(0,10,h.shape)
                w = np.random.binomial(10,(h+1)/11)
                y = np.round(np.clip(h + u - v + (w==3) + (w==4) + np.random.normal(0,self.noise,h.shape), 0,18)/3)
            else:
                y = np.round(np.clip(h + u + np.random.normal(0,self.noise,h.shape), 0,18)/3)
            
            u = np.clip(u/9, self.eps, 1-self.eps)
            y = np.clip(y/6, self.eps, 1-self.eps)
            if mode == 'multiple':
                huvwy = np.stack([h, u, v, w, y], 1)
                return torch.from_numpy(huvwy)
            else:
                huy = np.stack([h, u, y], 1)
                return torch.from_numpy(huy)
                
                
        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            get_vars(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            get_vars(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

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