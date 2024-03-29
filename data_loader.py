from __future__ import print_function, absolute_import
import math
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class VideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir,dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):
        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]
        num_ir = len(img_ir_paths)
        img_rgb_paths,pid_rgb,camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num_ir))
        if num_ir < S:  # 8 = chunk的数目，每个tracklet分成8段，每段随机选一帧
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_ir / S)
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:  # 8 = chunk的数目，每个tracklet分成8段，每段随机选一帧
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        if self.sample == 'video_train':

            idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])
            number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]
            imgs_ir   = []
            imgs_ir_f = []#添加！！！
            for index in number_ir:
                index = int(index)
                img_path   = img_ir_paths[index]
                img_path_f = '/data/lmh/datasets/fudiao/VCM_new/' + img_path[39:]#user



                img = read_image(img_path)
                img = np.array(img)
                # save_path = '/home/l/data_1/lmh/video_2/5.jpg'
                # cv2.imwrite(save_path, img)
                img_f = read_image(img_path_f)#!
                img_f = np.array(img_f)
                # save_path = '/home/l/data_1/lmh/video_2/6.jpg'
                # cv2.imwrite(save_path, img_f)

                if self.transform is not None:
                    img   = self.transform(img)
                    img_f = self.transform(img_f)#!
                imgs_ir.  append(img)
                imgs_ir_f.append(img_f)#!
            imgs_ir   = torch.cat(imgs_ir, dim=0)
            imgs_ir_f = torch.cat(imgs_ir_f, dim=0)#!

            ######################################################################################################################

            idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
            number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
            imgs_rgb   = []
            imgs_rgb_f = []#添加！！！
            for index in number_rgb:
                index = int(index)
                img_path   = img_rgb_paths[index]
                img_path_f = '/data/lmh/datasets/fudiao/VCM_new/' + img_path[39:]#user



                img = read_image(img_path)
                img = np.array(img)
                # save_path = '/home/l/data_1/lmh/video_2/7.jpg'
                # cv2.imwrite(save_path, img)
                img_f = read_image(img_path_f)#!
                img_f = np.array(img_f)#!
                # save_path = '/home/l/data_1/lmh/video_2/8.jpg'
                # cv2.imwrite(save_path, img_f)

                if self.transform is not None:
                    img   = self.transform(img)
                    img_f = self.transform(img_f)#!
                imgs_rgb.  append(img)
                imgs_rgb_f.append(img_f)#!
            imgs_rgb   = torch.cat(imgs_rgb, dim=0)
            imgs_rgb_f = torch.cat(imgs_rgb_f, dim=0)#!

            return imgs_ir,  imgs_ir_f,  pid_ir,  camid_ir, \
                   imgs_rgb, imgs_rgb_f, pid_rgb, camid_rgb

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class VideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:  # 8 = chunk的数目，每个tracklet分成8段，每段随机选一帧
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]
            imgs_ir   = []
            imgs_ir_f = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img_path_f = '/data/lmh/datasets/fudiao/VCM_new/' + img_path[39:]#user



                img = read_image(img_path)
                img = np.array(img)
                # save_path = '/home/l/data_1/lmh/video_2/5.jpg'
                # cv2.imwrite(save_path, img)
                img_f = read_image(img_path_f)
                img_f = np.array(img_f)
                # save_path = '/home/l/data_1/lmh/video_2/6.jpg'
                # cv2.imwrite(save_path, img_f)

                if self.transform is not None:
                    img   = self.transform(img)
                    img_f = self.transform(img_f)
                imgs_ir.  append(img)
                imgs_ir_f.append(img_f)
            imgs_ir   = torch.cat(imgs_ir, dim=0)
            imgs_ir_f = torch.cat(imgs_ir_f, dim=0)
            return imgs_ir, imgs_ir_f, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VideoDataset_train_evaluation(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, seq_len=12, sample='evenly', transform=None):
        self.dataset_ir = dataset_ir
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset_ir)

    def __getitem__(self, index):
        # print('len of dataset_ir = {}, now idx = {}'.format(len(self.dataset_ir),index))
        #print('index = ')
        #print(index)
        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[index]

        num_ir = len(img_ir_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            imgs_ir = []
            for index in indices:
                index=int(index)
                img_path = img_ir_paths[index]
                img = read_image(img_path)
                # 添加
                img = np.array(img)
                if self.transform is not None:  #这里还没看
                    img = self.transform(img)
                #img = img.unsqueeze(0)
                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            #imgs=imgs.permute(1,0,2,3)

            return imgs_ir, pid_ir, camid_ir
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

