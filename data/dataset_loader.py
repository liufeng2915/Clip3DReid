
import os
import torch
import numpy as np
import functools
import os.path as osp
import scipy.io
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = ToTensor()(img)
            # Calculate padding
            C, H, W = img_tensor.shape

            if H == W:
                padded_img_tensor = img_tensor
            else:
                diff = abs(H - W)
                padding_left = padding_right = diff // 2
                padding_top = padding_bottom = diff - diff // 2

                # Determine which dimension to pad
                if H < W:
                    padding = (0, 0, padding_left, padding_right)  # Pad the height
                else:
                    padding = (padding_top, padding_bottom, 0, 0)

                # Pad the image
                padded_img_tensor = F.pad(img_tensor, padding, 'constant', 0)  # Zero-padding

            pil_img = ToPILImage()(padded_img_tensor)
            #print(padded_img_tensor.shape)
            got_img = True

            # #
            mat_file = scipy.io.loadmat(img_path.split('.')[0]+'.mat')
            clip_feat = mat_file['clip_feat']
            if 'betas' in mat_file:
                betas = mat_file['betas']  # Now you can safely access 'betas'
            else:
                betas = np.zeros((1,10), dtype=np.float32)

        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return pil_img, clip_feat, betas


def read_test_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = ToTensor()(img)
            # Calculate padding
            C, H, W = img_tensor.shape

            if H == W:
                padded_img_tensor = img_tensor
            else:
                diff = abs(H - W)
                padding_left = padding_right = diff // 2
                padding_top = padding_bottom = diff - diff // 2

                # Determine which dimension to pad
                if H < W:
                    padding = (0, 0, padding_left, padding_right)  # Pad the height
                else:
                    padding = (padding_top, padding_bottom, 0, 0)

                # Pad the image
                padded_img_tensor = F.pad(img_tensor, padding, 'constant', 0)  # Zero-padding

            pil_img = ToPILImage()(padded_img_tensor)
            #print(padded_img_tensor.shape)
            got_img = True

        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return pil_img

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img, clip_feat, betas = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, clip_feat, betas, pid, camid, clothes_id

class TestImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_test_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id



class TrainImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, num_instances=4):
        self.dataset = dataset
        self.transform = transform

        self.fnames = []
        self.pids = []
        self.ret = []
        self.preprocess_img_path()
        self.num_data = int(len(self.fnames))
        self.upids = np.unique(self.pids)
        self.num_classes = len(self.upids)
        self.num_instances = num_instances

    def preprocess_img_path(self, relabel=True):

        all_pids = {}
        for (fpath, pid, cam, clothes_id) in self.dataset:
            fname = os.path.basename(fpath)
            self.fnames.append(fpath)

            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid

            pid = all_pids[pid]
            self.pids.append(pid)
            self.ret.append((fname, pid, cam, clothes_id))

    def __len__(self):
        return self.num_classes

    def __getitem__(self, index):
        pid = self.upids[index]
        img_idx = np.where(pid==self.pids)[0]
        if len(img_idx)<self.num_instances:
            img_idx = img_idx[np.random.choice(len(img_idx), self.num_instances)]
        else:
            img_idx = img_idx[np.random.permutation(len(img_idx))[:self.num_instances]]

        # image and identity label
        images = [self.transform(read_image(self.fnames[i])) for i in img_idx]
        img = torch.cat([images[i].unsqueeze(0) for i in range(self.num_instances)])

        label = np.array(self.pids)[img_idx]
        cam_id = np.array([self.ret[i][2] for i in img_idx])

        return img, label, cam_id

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.cloth_changing = cloth_changing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset[index]
        else:
            img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if self.cloth_changing:
            return clip, pid, camid, clothes_id
        else:
            return clip, pid, camid