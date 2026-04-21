import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch
from torch.utils.data import Sampler
import math
# several data augumentation strategies
def cv_random_flip(img, label, depth):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth


def randomCrop(image, label, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region)


def randomRotation(image, label, depth):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
    return image, label, depth



def cv_random_flip_two(img, label):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop_two(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation_two(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize,
                 condition_size=(512, 512), target_size=(512, 512), condition_type: str = "canny",
                 drop_text_prob: float = 0.1, drop_image_prob: float = 0.1, return_pil_image: bool = False,
                 position_scale=1.0):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')
                    or f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

        self.position_scale = position_scale
        self.condition_type = condition_type


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])
        image, gt, depth = cv_random_flip(image, gt, depth)
        image, gt, depth = randomCrop(image, gt, depth)
        image, gt, depth = randomRotation(image, gt, depth)
        image = colorEnhance(image)
        # gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)

        position_delta = np.array([0, 0])
        description = "give the saliency map of the image"

        gt = gt.repeat(3, 1, 1) 
        # return {
        #     "image": gt,
        #     "condition_0": image,
        #     "position_delta_0": position_delta,
        #     "condition_1": depth,
        #     "position_delta_1": position_delta,
        #     "condition_type": self.condition_type,
        #     "description": description,
        #     # **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
        #     **({"position_scale_0": self.position_scale} if self.position_scale != 1.0 else {}),
        # }
    
        return {
                "image": gt,
                "condition_0": image,
                "position_delta_0": position_delta,
                "condition_type_0": self.condition_type,
                "description": description,
                # **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
                **({"position_scale_0": self.position_scale} if self.position_scale != 1.0 else {}),
            }

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            # print(img_path)
            # print(gt_path)
            # print(depth_path)
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            # if img.size == gt.size and gt.size == depth.size:
            images.append(img_path)
            gts.append(gt_path)
            depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                      Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size



class RGBSalObjDataset(data.Dataset):
    def __init__(self, datasets, imgsize, mode,
                 condition_size=(512, 512), target_size=(512, 512), condition_type: str = "canny",
                 drop_text_prob: float = 0.1, drop_image_prob: float = 0.1, return_pil_image: bool = False,
                 position_scale=1.0):
        self.imgsize = imgsize
        if type(datasets) != list:
            datasets = [datasets]
        self.datas_id = []
        self.mode = mode
        
        for (i, dataset) in enumerate(datasets):
            if mode == 'train':
                data_dir = '/root/user-data/dataset/shr_data/unisod/rgb_dataset/{}'.format(dataset)
                imgset_path = data_dir + '/train.txt'

            else:
                data_dir = '/root/user-data/dataset/shr_data/unisod/rgb_dataset/{}'.format(dataset)
                imgset_path = data_dir + '/test.txt'

            imgset_file = open(imgset_path)

            for line in imgset_file:
                data = {}
                img_path = line.strip("\n").split(" ")[0]
                gt_path = line.strip("\n").split(" ")[1]
                data['img_path'] = data_dir + img_path
                data['gt_path'] = data_dir + gt_path
                data['dataset'] = dataset
                self.datas_id.append(data)


        self.img_transform = transforms.Compose([
            transforms.Resize((self.imgsize, self.imgsize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if self.mode =="train":
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.imgsize, self.imgsize)),
                transforms.ToTensor()])
        else:
            self.gt_transform = transforms.Compose([
                transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.imgsize, self.imgsize)), transforms.ToTensor()])

        self.position_scale = position_scale
        self.condition_type = condition_type


    def __getitem__(self, index):

        assert os.path.exists(self.datas_id[index]['img_path']), (
            '{} does not exist'.format(self.datas_id[index]['img_path']))
        assert os.path.exists(self.datas_id[index]['gt_path']), (
            '{} does not exist'.format(self.datas_id[index]['gt_path']))

        image = Image.open(self.datas_id[index]['img_path']).convert('RGB')
        gt = Image.open(self.datas_id[index]['gt_path']).convert('L')
        if self.mode == 'train':
            image, gt = cv_random_flip_two(image, gt)
            image, gt = randomCrop_two(image, gt)
            image, gt = randomRotation_two(image, gt)
            image = colorEnhance(image)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        
        position_delta = np.array([0, 0])
        description = "RGB to mask"

        if self.mode == 'train':
            gt = gt.repeat(3, 1, 1)  

        sample = {
                "image": gt,
                "condition_0": image,
                "position_delta_0": position_delta,
                "condition_type_0": self.condition_type,
                "description": description,
                "dataset": self.datas_id[index]['dataset'],
                "name": self.datas_id[index]['gt_path'].split('/')[-1]
                
            }
        if self.position_scale != 1.0:
            sample["position_scale_0"] = self.position_scale
    
        return sample

    def __len__(self):
        return len(self.datas_id)



class RGBDSalObjDataset(data.Dataset):
    def __init__(self, datasets, imgsize, mode,
                 condition_size=(512, 512), target_size=(512, 512), condition_type: str = "canny",
                 drop_text_prob: float = 0.1, drop_image_prob: float = 0.1, return_pil_image: bool = False,
                 position_scale=1.0):
        self.imgsize = imgsize
        if type(datasets) != list:
            datasets = [datasets]
        self.datas_id = []
        self.mode = mode
        
        for (i, dataset) in enumerate(datasets):
            if mode == 'train':
                data_dir = '/root/user-data/dataset/shr_data/unisod/rgbd_dataset/{}'.format(dataset)
                imgset_path = data_dir + '/train.txt'

            else:
                data_dir = '/root/user-data/dataset/shr_data/unisod/rgbd_dataset/{}'.format(dataset)
                imgset_path = data_dir + '/test.txt'

            imgset_file = open(imgset_path)

            for line in imgset_file:
                data = {}
                img_path = line.strip("\n").split(" ")[0]
                gt_path = line.strip("\n").split(" ")[1]
                depth_path = line.strip("\n").split(" ")[2]
                data['img_path'] = data_dir + img_path
                data['gt_path'] = data_dir + gt_path
                data['depth_path'] = data_dir + depth_path
                data['dataset'] = dataset
                self.datas_id.append(data)


        self.img_transform = transforms.Compose([
            transforms.Resize((self.imgsize, self.imgsize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if self.mode =="train":
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.imgsize, self.imgsize)),
                transforms.ToTensor()])
        else:
            self.gt_transform = transforms.Compose([
                transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.imgsize, self.imgsize)), transforms.ToTensor()])

        self.position_scale = position_scale
        self.condition_type = condition_type


    def __getitem__(self, index):

        assert os.path.exists(self.datas_id[index]['img_path']), (
            '{} does not exist'.format(self.datas_id[index]['img_path']))
        assert os.path.exists(self.datas_id[index]['gt_path']), (
            '{} does not exist'.format(self.datas_id[index]['gt_path']))
        assert os.path.exists(self.datas_id[index]['depth_path']), (
                '{} does not exist'.format(self.datas_id[index]['depth_path']))

        image = Image.open(self.datas_id[index]['img_path']).convert('RGB')
        gt = Image.open(self.datas_id[index]['gt_path']).convert('L')
        depth = Image.open(self.datas_id[index]['depth_path']).convert('RGB')

        if self.mode == 'train':
            image, gt, depth = cv_random_flip(image, gt, depth)
            image, gt, depth = randomCrop(image, gt, depth)
            image, gt, depth = randomRotation(image, gt, depth)
            image = colorEnhance(image)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        position_delta = np.array([0, 0])
        description = "RGB and depth to mask"

        if self.mode == 'train':
            gt = gt.repeat(3, 1, 1) 
    


        sample = {
                "image": gt,
                "condition_0": image,
                "position_delta_0": position_delta,
                "condition_type_0": self.condition_type,
                "condition_1": depth,
                "position_delta_1": position_delta,
                "condition_type_1": self.condition_type,
                "description": description,
                "dataset": self.datas_id[index]['dataset'],
                "name": self.datas_id[index]['gt_path'].split('/')[-1]
                
            }
        if self.position_scale != 1.0:
            sample["position_scale_0"] = self.position_scale
            sample["position_scale_1"] = self.position_scale
    
        return sample

    def __len__(self):
        return len(self.datas_id)


class RGBTSalObjDataset(data.Dataset):
    def __init__(self, datasets, imgsize, mode,
                 condition_size=(512, 512), target_size=(512, 512), condition_type: str = "canny",
                 drop_text_prob: float = 0.1, drop_image_prob: float = 0.1, return_pil_image: bool = False,
                 position_scale=1.0):
        self.imgsize = imgsize
        if type(datasets) != list:
            datasets = [datasets]
        self.datas_id = []
        self.mode = mode
        
        for (i, dataset) in enumerate(datasets):
            if mode == 'train':
                data_dir = '/root/user-data/dataset/shr_data/unisod/rgbt_dataset/{}'.format(dataset)
                imgset_path = data_dir + '/train.txt'

            else:
                data_dir = '/root/user-data/dataset/shr_data/unisod/rgbt_dataset/{}'.format(dataset)
                imgset_path = data_dir + '/test.txt'

            imgset_file = open(imgset_path)

            for line in imgset_file:
                data = {}
                img_path = line.strip("\n").split(" ")[0]
                gt_path = line.strip("\n").split(" ")[1]
                thermal_path = line.strip("\n").split(" ")[2]
                data['img_path'] = data_dir + img_path
                data['gt_path'] = data_dir + gt_path
                data['thermal_path'] = data_dir + thermal_path
                data['dataset'] = dataset
                self.datas_id.append(data)


        self.img_transform = transforms.Compose([
            transforms.Resize((self.imgsize, self.imgsize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if self.mode =="train":
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.imgsize, self.imgsize)),
                transforms.ToTensor()])
        else:
            self.gt_transform = transforms.Compose([
                transforms.ToTensor()])
        self.thermals_transform = transforms.Compose(
            [transforms.Resize((self.imgsize, self.imgsize)), transforms.ToTensor()])

        self.position_scale = position_scale
        self.condition_type = condition_type


    def __getitem__(self, index):

        assert os.path.exists(self.datas_id[index]['img_path']), (
            '{} does not exist'.format(self.datas_id[index]['img_path']))
        assert os.path.exists(self.datas_id[index]['gt_path']), (
            '{} does not exist'.format(self.datas_id[index]['gt_path']))
        assert os.path.exists(self.datas_id[index]['thermal_path']), (
                '{} does not exist'.format(self.datas_id[index]['thermal_path']))

        image = Image.open(self.datas_id[index]['img_path']).convert('RGB')
        gt = Image.open(self.datas_id[index]['gt_path']).convert('L')
        thermal = Image.open(self.datas_id[index]['thermal_path']).convert('RGB')

        if self.mode == 'train':
            image, gt, thermal = cv_random_flip(image, gt, thermal)
            image, gt, thermal = randomCrop(image, gt, thermal)
            image, gt, thermal = randomRotation(image, gt, thermal)
            image = colorEnhance(image)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        thermal = self.thermals_transform(thermal)
        position_delta = np.array([0, 0])
        description = "RGB and thermal to mask"

        if self.mode == 'train':
            gt = gt.repeat(3, 1, 1) 


        sample = {
                "image": gt,
                "condition_0": image,
                "position_delta_0": position_delta,
                "condition_type_0": self.condition_type,
                "condition_1": thermal,
                "position_delta_1": position_delta,
                "condition_type_1": self.condition_type,
                "description": description,
                "dataset": self.datas_id[index]['dataset'],
                "name": self.datas_id[index]['gt_path'].split('/')[-1]
                
            }
        if self.position_scale != 1.0:
            sample["position_scale_0"] = self.position_scale
            sample["position_scale_1"] = self.position_scale
    
        return sample

    def __len__(self):
        return len(self.datas_id)

            

    def __len__(self):
        return len(self.datas_id)

class MultiDataset(data.Dataset):
    def __init__(self, datasets: dict):
        self.datasets = datasets
        self.names = list(datasets.keys())
        self.offsets = {}
        offset = 0
        for n in self.names:
            self.offsets[n] = offset
            offset += len(datasets[n])
        self.total_len = offset

    def __len__(self):
        return self.total_len

    def __getitem__(self, global_idx):
        for n in self.names:
            start = self.offsets[n]
            end = start + len(self.datasets[n])
            if start <= global_idx < end:
                local_idx = global_idx - start
                sample = self.datasets[n][local_idx]

                return sample, n
        raise IndexError


class MultiSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        datasets: dict,
        batch_size: int,
        weights: dict = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.names = list(datasets.keys())
        self.batch_size = batch_size

        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

        # offsets（global index）
        self.offsets = {}
        offset = 0
        for name in self.names:
            self.offsets[name] = offset
            offset += len(datasets[name])

        # weights
        if weights is None:
            weights = {n: 1.0 for n in self.names}
        w = torch.tensor([weights[n] for n in self.names], dtype=torch.float)
        self.probs = (w / w.sum())
        

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _shuffle_dataset(self, name, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(len(self.datasets[name]), generator=g)

        if self.world_size > 1:
            idx = idx[self.rank::self.world_size]

        return idx.tolist()

    def __iter__(self):
        refresh_count = {n: 0 for n in self.names}

        def build_seed(name):
            # 不同 epoch、不同 dataset、不同 refresh 都有不同随机流
            return self.seed + self.epoch * 100000 + self.names.index(name) * 1000 + refresh_count[name]

        indices = {}
        ptrs = {}
        for n in self.names:
            indices[n] = self._shuffle_dataset(n, build_seed(n))
            ptrs[n] = 0

        while True:
            task_idx = torch.multinomial(self.probs, 1).item()
            task = self.names[task_idx]
            batch = []

            for _ in range(self.batch_size):
                if ptrs[task] >= len(indices[task]):
                    refresh_count[task] += 1
                    indices[task] = self._shuffle_dataset(task, build_seed(task))
                    ptrs[task] = 0

                local_idx = indices[task][ptrs[task]]
                global_idx = self.offsets[task] + local_idx
                batch.append(global_idx)
                ptrs[task] += 1

            yield batch




class DDPSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        datasets: dict,
        batch_size: int,
        weights: dict = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        total_steps: int = 1000000,   # ⭐ 控制训练长度
    ):
        self.datasets = datasets
        self.names = sorted(datasets.keys())
        self.batch_size = batch_size

        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.total_steps = total_steps

        # offsets（global index）
        self.offsets = {}
        offset = 0
        for name in self.names:
            self.offsets[name] = offset
            offset += len(datasets[name])

        # weights
        if weights is None:
            weights = {n: 1.0 for n in self.names}
        w = torch.tensor([weights[n] for n in self.names], dtype=torch.float)
        self.probs = (w / w.sum())

    def set_epoch(self, epoch):
        self.epoch = epoch

    # 🔥 DDP安全shuffle（长度完全一致）
    def _shuffle_dataset(self, name, g):
        full = torch.randperm(len(self.datasets[name]), generator=g)

        if self.world_size > 1:
            total = len(full)
            pad = (self.world_size - total % self.world_size) % self.world_size

            if pad > 0:
                full = torch.cat([full, full[:pad]])

            full = full.view(self.world_size, -1)
            idx = full[self.rank]
        else:
            idx = full

        return idx.tolist()

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 🔥 所有GPU共享完全一致的task schedule
        schedule = torch.multinomial(
            self.probs,
            num_samples=self.total_steps,
            replacement=True,
            generator=g,
        ).tolist()

        # 初始化各dataset
        indices = {n: self._shuffle_dataset(n, g) for n in self.names}
        ptrs = {n: 0 for n in self.names}

        for task_idx in schedule:
            task = self.names[task_idx]

            batch = []
            for _ in range(self.batch_size):
                if ptrs[task] >= len(indices[task]):
                    # 🔥 用完立即reshuffle（各GPU同步）
                    indices[task] = self._shuffle_dataset(task, g)
                    ptrs[task] = 0

                local_idx = indices[task][ptrs[task]]
                global_idx = self.offsets[task] + local_idx

                batch.append(global_idx)
                ptrs[task] += 1

            yield batch

    def __len__(self):
        return self.total_steps








class SalObjDataset_val(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize,
                 condition_size=(512, 512), target_size=(512, 512), condition_type: str = "canny",
                 drop_text_prob: float = 0.1, drop_image_prob: float = 0.1, return_pil_image: bool = False,
                 position_scale=1.0):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')
                    or f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

        self.position_scale = position_scale
        self.condition_type = condition_type


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])

        image_original = image
        gt_original = gt

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)

        position_delta = np.array([0, 0])
        description = "give the saliency map of the image"
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        return {
            "image": gt,
            "condition_0": image,
            "position_delta_0": position_delta,
            "condition_1": depth,
            "position_delta_1": position_delta,
            "condition_type": self.condition_type,
            "description": description,
            "name": name,
            # **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale_0": self.position_scale} if self.position_scale != 1.0 else {}),
        }
        

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            # print(img_path)
            # print(gt_path)
            # print(depth_path)
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            # if img.size == gt.size and gt.size == depth.size:
            images.append(img_path)
            gts.append(gt_path)
            depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                      Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size





###############################################################################
# 0919
#

class SalObjDataset_var(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize):

        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

    def __getitem__(self, index):

        ## read imag, gt, depth
        image0 = self.rgb_loader(self.images[index])
        gt0 = self.binary_loader(self.gts[index])
        depth0 = self.rgb_loader(self.depths[index])

        ##################################################
        ## out1
        ##################################################
        image, gt, depth = cv_random_flip(image0, gt0, depth0)
        image, gt, depth = randomCrop(image, gt, depth)
        image, gt, depth = randomRotation(image, gt, depth)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)

        ##################################################
        ## out1
        ##################################################
        image2, gt2, depth2 = cv_random_flip(image0, gt0, depth0)
        image2, gt2, depth2 = randomCrop(image2, gt2, depth2)
        image2, gt2, depth2 = randomRotation(image2, gt2, depth2)
        image2 = colorEnhance(image2)
        gt2 = randomPeper(gt2)
        image2 = self.img_transform(image2)
        gt2 = self.gt_transform(gt2)
        depth2 = self.depths_transform(depth2)

        return image, gt, depth, image2, gt2, depth2

    def filter_files(self):

        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            if img.size == gt.size and gt.size == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                      Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size


class SalObjDataset_var_unlabel(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize):

        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

    def __getitem__(self, index):

        ## read imag, gt, depth
        image0 = self.rgb_loader(self.images[index])
        gt0 = self.binary_loader(self.gts[index])
        depth0 = self.binary_loader(self.depths[index])

        ##################################################
        ## out1
        ##################################################
        image, gt, depth = cv_random_flip(image0, gt0, depth0)
        image, gt, depth = randomCrop(image, gt, depth)
        image, gt, depth = randomRotation(image, gt, depth)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)

        ##################################################
        ## out1
        ##################################################
        image2, gt2, depth2 = cv_random_flip(image0, gt0, depth0)
        image2, gt2, depth2 = randomCrop(image2, gt2, depth2)
        image2, gt2, depth2 = randomRotation(image2, gt2, depth2)
        image2 = colorEnhance(image2)
        gt2 = randomPeper(gt2)
        image2 = self.img_transform(image2)
        gt2 = self.gt_transform(gt2)
        depth2 = self.depths_transform(depth2)

        return image, gt, depth, image2, gt2, depth2

    def filter_files(self):

        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            if img.size == gt.size and gt.size == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                      Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):
    dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize)
    print(len(dataset))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# dataloader for training2
## 09-19-2020
def get_loader_var(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=12,
                   pin_memory=False):
    dataset = SalObjDataset_var(image_root, gt_root, depth_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def get_loader_var_unlabel(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=12,
                           pin_memory=False):
    dataset = SalObjDataset_var_unlabel(image_root, gt_root, depth_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.rgb_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
