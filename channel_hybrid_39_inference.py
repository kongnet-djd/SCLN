import json
import pickle
import os
import copy
import cv2
import glob
import random
import numpy as np
from PIL import Image, ImageFile
import torch
import pandas as pd
import tifffile
from torchvision import transforms
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from metric import metrics
import time

class BasicBlock(nn.Module):
    expansion = 1  # 每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=(3, 3, 1), stride=(stride, stride, 1), padding=(1, 1, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)  # BN处理
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Encoding(nn.Module):
    def __init__(self, block, blocks_num, include_top=True):  # block残差结构 include_top为了之后搭建更加复杂的网络
        super(Encoding, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(7, 7, 1), stride=(2, 2, 1), padding=(3, 3, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=2)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, channel * block.expansion, kernel_size=(1, 1, 1), stride=(stride, stride, 1),
                          bias=False),
                nn.BatchNorm3d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        l1 = self.conv1(x)
        l2 = self.layer1(l1)
        l3 = self.layer2(l2)
        l4 = self.layer3(l3)
        l5 = self.layer4(l4)
        return l1, l2, l3, l4, l5


# %%
class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # (2, 3, 256, 14, 14) -- (2, 3, 1, 1, 1) --只能按照第一个维度压缩
        self.fc = nn.Sequential(
            nn.Linear(channel, 128, bias=False),
            #             nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, channel, bias=False),
            #             nn.Dropout(0.5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x.permute(0, 4, 2, 3, 1)
        b, c, _, _, _ = x1.size()

        y = self.avg_pool(x1).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1, 1)

        z = x1 * y.expand_as(x1)
        z1 = z.permute(0, 4, 2, 3, 1)

        return z1


# %%
class Channel_hybrid(nn.Module):

    def __init__(self, in_channels, feature_scale=1, n_classes=1, is_deconv=True, nonlocal_mode='concatenation',
                 attention_dsample=(1, 1, 1), is_batchnorm=True):
        super(Channel_hybrid, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        self.encoding = Encoding(BasicBlock, [2, 2, 2, 2])
        self.se = SELayer(channel=39)

        # upsampling
        self.up_concat4 = UnetUp3(512, 256, d_state=512, is_layer5=True)
        self.up_concat3 = UnetUp3(256, 128, d_state=256, is_layer5=False)
        self.up_concat2 = UnetUp3(128, 64, d_state=128, is_layer5=False)
        self.up_concat1 = UnetUp3(128, 64, d_state=64, is_layer5=False)

        self.up = nn.ConvTranspose3d(64, 64, kernel_size=(4, 4, 3), stride=(2, 2, 1), padding=(1, 1, 1))
        self.conv1 = nn.Sequential(nn.Conv3d(64, 32, (3, 3, 7), (1, 1, 2), (1, 1, 1)),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv3d(32, 64, (3, 3, 7), (1, 1, 2), (1, 1, 1)),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU(inplace=True), )
        self.conv3 = nn.Conv3d(64, 1, (3, 3, 7), (1, 1, 2), (1, 1, 0))

        #         self.up_concat0 = UnetUp3(2, 1, d_state=64, is_layer5= False)

        self.final1 = nn.Sigmoid()

    def forward(self, inputs):
        # Feature Extraction

        d_1, d_2, d_3, d_4, d_5 = self.encoding(inputs)

        se1 = self.se(d_1)
        se2 = self.se(d_2)
        se3 = self.se(d_3)
        se4 = self.se(d_4)
        se5 = self.se(d_5)

        up4 = self.up_concat4(se4, se5)  # (-1,256,14,14,1)
        up3 = self.up_concat3(se3, up4)  # (-1,128,28,28,c)
        up2 = self.up_concat2(se2, up3)  # (-1, 64,56,56,c)
        up1 = self.up_concat1(se1, up2)  # (-1,64, 112, 112, c)
        up0 = self.up(up1)
        up0_1 = self.conv1(up0)
        up0_2 = self.conv2(up0_1)
        up0_3 = self.conv3(up0_2)

        final1 = self.final1(up0_3)

        return up0_3, final1


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 39), padding_size=(1, 1, 1),
                 init_stride=(1, 1, 1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            #             self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, (3,3), (1,1), (1,1)),
            #                                        nn.BatchNorm2d(out_size),
            #                                        nn.ReLU(inplace=True),)
            #             self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, (3,3), (1,1), (1,1)),
            #                                        nn.BatchNorm2d(out_size),
            #                                        nn.ReLU(inplace=True),)

            #             # 如果把第二个卷积核的size 改了的话，在up cov的时候又变成了2d，还是hybrid
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        # (b, d, 224,224, 1)
        return outputs


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, d_state, is_layer5, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        self.conv = UnetConv3(in_size, out_size, is_batchnorm)
        self.up = nn.ConvTranspose3d(d_state, out_size, kernel_size=(4, 4, 3), stride=(2, 2, 1),
                                     padding=(1, 1, 1))  # 共享参数

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)  # 因为第二个是上采样右边的输出，无论怎样，输出的维度都是 1
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]

        outputs1 = F.pad(inputs1, padding)  # 这时inputs变成2维了

        cat_result = torch.cat([outputs1, outputs2], 1)
        cov_result = self.conv(cat_result)

        return cov_result


def dice_coeff(pred, target):
    # smooth = 1.
    smooth = 1e-5
    num = pred.shape[0]
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# %%
class DataPreparation(object):
    root_path = "/"
    tiff_rootPath = os.path.join(root_path, 'mnt')

    #     tiff_path1 = os.path.join('/mnt', 'ome')
    #     mask_path1 = os.path.join('/mnt', 'Basel_Zuri_masks')

    def __init__(self, gt_type='grade'):
        self.gt_type = gt_type
        self.valid_channelList = []
        self.split_list = dict()
        self.percentile_minmax = dict()
        #         self.tiffPath_gtLabelList = []
        self.tiffPath_gtLabelList_brca = []

    #         self.tiff_path1 = os.path.join(self.tiff_rootPath, 'ome')
    #         self.mask_path1 = os.path.join(self.tiff_rootPath, 'Basel_Zuri_masks')

    def split_dataset(self):
        print("Computing new split...")
        ###
        ### brca_metadata
        #         brca = pd.read_csv('/mnt/breast_metadata/metabric_img/idr0076/brca_gt.csv', usecols=['METABRIC.ID','Grade'],index_col=None)
        #         brca_dict = brca.to_dict()
        #         info_list = list(brca_dict['METABRIC.ID'].values())
        #         grade_list = list(brca_dict['Grade'].values())

        real_path = '/home/HDD-2T-2023/djd/to_public_repository/testsets/full_stacks/'
        real_name = []
        for i in sorted(os.listdir(real_path)):
            if i.split('.')[1] == 'tiff':
                a = i.split('.')[0]
                b = a[:-9]
                c = b + 'cellmask.tiff'
                tiff_path = '/home/HDD-2T-2023/djd/to_public_repository/testsets/full_stacks/' + i
                mask_path = '/home/HDD-2T-2023/djd/to_public_repository/testsets/cell_masks/' + c
                #             if b in info_list:
                #                 idx = info_list.index(b)
                #                 grade = grade_list[idx]-1
                self.tiffPath_gtLabelList_brca.append({'tiff_path': tiff_path, 'mask_path': mask_path})
        #                 self.total.append({'tiff_path': tiff_path,
        #                                    'class_label': grade})
        #         print(self.tiffPath_gtLabelList)

        total_num_imgs = len(self.tiffPath_gtLabelList_brca)
        num_trainData = int(total_num_imgs * 0.7)
        self.split_list['train'] = self.tiffPath_gtLabelList_brca[:num_trainData]
        self.split_list['test'] = self.tiffPath_gtLabelList_brca[0:]
        print("Num training imgs: ", len(self.split_list['train']))
        print("Num testing imgs: ", len(self.split_list['test']))

    def get_channelList(self, channels_toExclude=[]):
        # 39 channel
        csv_path = 'breastcancer_39.xlsx'
        df = pd.read_excel(csv_path, header=None, names=['Channel Names'])
        listChannels = df['Channel Names'].values.tolist()
        print('Channels used for training and test')
        for channel_id, channel_name in enumerate(listChannels):
            if channel_name not in channels_toExclude:
                #                 print(channel_name)
                self.valid_channelList.append({'channel_id': channel_id,
                                               'channel_name': channel_name})
        print('')

    def find_percentile(self):
        print('Finding max intensities across all images for each channel (min is always 0)')
        channel_maxvals = np.zeros(len(self.valid_channelList))
        for img_id, tiffPath_gtLabel in enumerate(self.tiffPath_gtLabelList_brca):
            for channel_id, valid_channel in enumerate(self.valid_channelList):
                channel_img = tifffile.imread(tiffPath_gtLabel['tiff_path'],
                                              key=valid_channel['channel_id'])
                max_val = np.max(channel_img)
                if channel_maxvals[channel_id] < max_val:
                    channel_maxvals[channel_id] = max_val
        print(channel_maxvals)
        print('')
        print('Finding min and max percentiles across all images for each channel')
        num_bins = 10000
        channel_histCum = np.zeros((len(self.valid_channelList), num_bins - 2), np.int64)
        for img_id, tiffPath_gtLabel in enumerate(self.tiffPath_gtLabelList_brca):
            # print('{}/{}'.format(img_id, len(self.tiffPath_gtLabelList)))
            #             tiff_reader = bioformats.ImageReader(os.path.join(self.tiff_rootPath,
            #                                                               tiffPath_gtLabel['tiff_path']))
            for channel_id, valid_channel in enumerate(self.valid_channelList):
                #                 channel_img = tiff_reader.read(index=valid_channel['channel_id'],
                #                                                rescale=False)
                channel_img = tifffile.imread(tiffPath_gtLabel['tiff_path'],
                                              key=valid_channel['channel_id'])
                bin_list = np.arange(0, channel_maxvals[channel_id] - 1e-6,
                                     channel_maxvals[channel_id] / num_bins)
                channel_hist = np.histogram(channel_img.flatten(), bin_list)[0]
                # excluding bin 0 because they are too many background pixels
                channel_histCum[channel_id] += channel_hist[1:]
        for channel_id, channel_hist in enumerate(channel_histCum):
            cum_sum = channel_hist.cumsum()
            one_percent = cum_sum[-1] * 0.01
            max_id = np.where(cum_sum > (cum_sum[-1] - one_percent))[0][0]
            bin_list = np.arange(0, channel_maxvals[channel_id],
                                 channel_maxvals[channel_id] / num_bins)
            max_val = bin_list[max_id] + channel_maxvals[channel_id] / num_bins
            self.valid_channelList[channel_id]['max_percentile'] = max_val
            print(max_val)
        print('')


# %%
class BioformatTiffreader(object):
    root_path = ""

    #     tiff_rootPath = os.path.join(root_path, 'mnt/ome')
    #     tiff_path1 = os.path.join('/mnt', 'ome')
    #     mask_path1 = os.path.join('/mnt', 'Basel_Zuri_masks')

    def __init__(self, prepared_data, img_size=224, phase_str='train'):
        print('Creating tiff readers...')
        self.tiffPath_gtLabels = prepared_data.split_list[phase_str]
        self.valid_channelList = prepared_data.valid_channelList

        # Input size for a network
        self.img_size = img_size
        # self.tiffPath_gtLabels = self.tiffPath_gtLabels[:16]
        self.normalize = transforms.Normalize(mean=[0.5],
                                              std=[0.25])
        if phase_str == 'train':
            # Training Set
            self.transforms = transforms.Compose([
                #                 transforms.RandomRotation((-90, 90)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.CenterCrop(img_size),
                #                 transforms.RandomCrop(img_size, pad_if_needed=True),
                #                 transforms.Resize(img_size),
                transforms.ToTensor(),
                #                 transforms.ColorJitter(brightness=0.13, contrast=0.13,
                #                                        saturation=0.13, hue=0.13),
                #                 transforms.RandomErasing(p=0.18, scale=(0.02, 0.25), ratio=(0.5, 25),
                #                                          value=0.5, inplace=False)
            ])
        else:
            # Test Set
            # if img size is larger than 224
            self.transforms = transforms.Compose([transforms.CenterCrop(img_size),
                                                  transforms.ToTensor(), ])

    #             # Test Set
    #             # if img size is smaller than 224
    #             self.transforms = transforms.Compose([transforms.Resize(img_size, img_size),
    # #                                                   transforms.CenterCrop(img_size),
    #                                                   transforms.ToTensor(),])

    def __getitem__(self, idx):
        tiff_path = self.tiffPath_gtLabels[idx]['tiff_path']
        mask_path = self.tiffPath_gtLabels[idx]['mask_path']
        #         class_label = self.tiffPath_gtLabels[idx]['class_label']

        in_img = np.zeros((len(self.valid_channelList), self.img_size, self.img_size), np.float32)
        #         tiff_reader = bioformats.ImageReader(os.path.join(self.tiff_rootPath, tiff_path))
        tensor_imgList = []
        #         seed = np.random.randint(541474)  # make a seed with numpy generator
        for chId_iter, valid_channel in enumerate(self.valid_channelList):
            in_channel = tifffile.imread(tiff_path,
                                         key=valid_channel['channel_id'])
            #             in_channel = tiff_reader.read(index=valid_channel['channel_id'], rescale=False)
            min_val = 0
            max_val = valid_channel['max_percentile']
            # 这一步应该是去除掉特别亮的点，就是intensity特别大的点
            in_channel = (np.clip(in_channel, min_val, max_val) - min_val) / (max_val - min_val)
            # 这个意思的是，如果输入图片的尺寸太瘦或者太高，那么就填充一下尺寸到，224
            if in_channel.shape[0] < self.img_size:
                pad_len = int(self.img_size - in_channel.shape[0])
                in_channel = np.pad(in_channel, ((0, 0), (pad_len, 0)), 'constant', constant_values=(0, 0))
            if in_channel.shape[1] < self.img_size:
                pad_len = int(self.img_size - in_channel.shape[1])
                in_channel = np.pad(in_channel, ((0, 0), (0, pad_len)), 'constant', constant_values=(0, 0))
            pil_img = Image.fromarray((255 * in_channel).astype(np.uint8))
            #             random.seed(seed)  # apply this seed to img tranfsorms
            tensor_img = self.transforms(pil_img)
            #             tensor_img = self.normalize(tensor_img)
            tensor_imgList.append(tensor_img)
        in_img = torch.cat(tensor_imgList)

        #         in_mask = np.zeros((self.img_size, self.img_size), np.float32)
        #         tiff_reader = bioformats.ImageReader(os.path.join(self.tiff_rootPath, tiff_path)

        mask = tifffile.imread(mask_path)
        mask[mask > 0] = 1

        # 这个意思的是，如果输入图片的尺寸太瘦或者太高，那么就填充一下尺寸到，224
        if mask.shape[0] < self.img_size:
            pad_len = int(self.img_size - mask.shape[0])
            mask = np.pad(mask, ((0, 0), (pad_len, 0)), 'constant', constant_values=(0, 0))
        if mask.shape[1] < self.img_size:
            pad_len = int(self.img_size - mask.shape[1])
            mask = np.pad(mask, ((0, 0), (0, pad_len)), 'constant', constant_values=(0, 0))
        pil_mask = Image.fromarray((255 * mask).astype(np.uint8))
        #         pil_mask_array =
        #         random.seed(seed)  # apply this seed to img tranfsorms
        tensor_mask = self.transforms(pil_mask)
        #         tensor_mask = self.normalize(tensor_mask)

        return in_img, tensor_mask, tiff_path

    def __len__(self):
        return len(self.tiffPath_gtLabels)


# %%
def create_dataloader(dataset_name, gt_type, img_size,
                      batch_size=4, recompute_dataset=False):
    if dataset_name == "cytometry":
        return create_cytometry(gt_type, img_size, batch_size, recompute_dataset)
    # elif dataset_name == 'synthetic':
    #     return create_synthetic(batch_size)
    else:
        print('wrong dataset_name')
    # %%


def create_cytometry(gt_type, img_size, batch_size, recompute_dataset):
    num_classes = 1
    # Path to the prepared data
    prepared_dataPath = 'mmmm_39.pkl'
    channels_toExclude = ['not_an_antigen']
    # '|' 按位或运算符：只要对应的二个二进位有一个为1时，结果位就为1
    if recompute_dataset | (os.path.exists(prepared_dataPath) == False):
        # define a class for data preparation
        prepared_data = DataPreparation(gt_type=gt_type)
        # resplit the dataset into train/test
        prepared_data.split_dataset()
        # recompute list of channels
        prepared_data.get_channelList(channels_toExclude)
        # recompute percentile
        prepared_data.find_percentile()
        # Save the data preparation obj
        with open(prepared_dataPath, 'wb') as pkl_file:
            pickle.dump(prepared_data, pkl_file)
    else:
        # load the data preparation obj
        with open(prepared_dataPath, 'rb') as pkl_file:
            prepared_data = pickle.load(pkl_file)
    # dataset_train = BioformatTiffreader(prepared_data,
    #                                     img_size=img_size,
    #                                     phase_str='train')
    dataset_test = BioformatTiffreader(prepared_data,
                                       img_size=img_size,
                                       phase_str='test')
    dataset_loader = dict()
    # dataset_loader['train'] = torch.utils.data.DataLoader(
    #     dataset=dataset_train, batch_size=batch_size, shuffle=True,
    #     drop_last=True, num_workers=3)
    dataset_loader['test'] = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=3)
    num_channels = len(dataset_test.valid_channelList)
    phase_list = ['train', 'test']
    return dataset_loader, num_channels, num_classes, phase_list




#%%
dataset_name = 'cytometry'
method = 'ours-2d-shared'
gt_type = 'grade'
# torch.cuda.set_device(0)
# Hyperparameters for training
# learning_rate = 0.0001
# lrrate_str = 'lr_{}'.format(str(learning_rate).split('.')[-1])#"lr_0001"
# print("learning_rate: {}".format(learning_rate))
img_size = 224
print("img_size: ", img_size)
num_epochs = 200
print("num_epochs: ", num_epochs)
# batch_size = 8*torch.cuda.device_count()
# 这是自己改的
batch_size = 1
print("Training {} on {} using {}".format(gt_type, dataset_name, method))

# Dataloader
dataset_loader, num_channels, num_classes, phase_list = create_dataloader(
    dataset_name, gt_type, img_size,
    batch_size=batch_size, recompute_dataset=False)

device = torch.device("cuda:0")  # PyTorch v0.4.0

model = Channel_hybrid(in_channels=18)


# 计算模型参数的总数
total_params = sum(p.numel() for p in model.parameters())
memory_usage_bytes = total_params * 4  # float32 占用4字节
memory_usage_mb = memory_usage_bytes / (1024 ** 2)
print(f"模型参数总数: {total_params}")
print(f"大约内存占用: {memory_usage_mb} MB")

model = torch.load('cean39_model/best38.pt', map_location=device)

# Send the model to GPU
# model.to(device)


# Start training
print(" START TIME: ", time.ctime())
since = time.time()


dataloader = dataset_loader['test']
running_dice = []
running_sens = []
running_spec = []
running_rec = []
running_prec = []
running_acc = []
running_jaccard = []
counter = 0
for x, y, tiff_path in dataloader:
    kk = x.permute(0, 2, 3, 1)
    kk1 = kk.unsqueeze(1)
    labels = y.unsqueeze(4)
    inputs = kk1.to(device)
    labels = labels.to(device)

    pred, pred_dice = model(inputs)
    pred_dice[pred_dice >= 0.6] = 1
    pred_dice[pred_dice < 0.6] = 0
    # pred_dice = pred_dice*255
     # label1 = label.detach().numpy()

    name = str(tiff_path).split('.')[0].strip('(\'').split('/')[-1] + '.npy'
    possibility = torch.squeeze(pred_dice).cpu().detach().numpy()
    # np.save(os.path.join('npy_save/cean39', name), possibility)

    dice = dice_coeff(pred_dice, labels)
    sens = metrics.sensitivity(pred_dice, labels)
    spec = metrics.specificity(pred_dice, labels)
    rec = metrics.recall(pred_dice, labels)
    prec = metrics.precision(pred_dice, labels)
    acc = metrics.accuracy(pred_dice, labels)
    jaccard = metrics.jaccard_index(pred_dice, labels)

    running_dice.append(dice.item())
    running_sens.append(sens.item())
    running_spec.append(spec.item())
    running_rec.append(rec.item())
    running_prec.append(prec.item())
    running_acc.append(acc.item())
    running_jaccard.append(jaccard.item())
    counter += 1

avg_dice = np.average(running_dice)
avg_sens = np.average(running_sens)
avg_spec = np.average(running_spec)
avg_rec = np.average(running_rec)
avg_prec = np.average(running_prec)
avg_acc = np.average(running_acc)
avg_jaccard = np.average(running_jaccard)

std_dice = np.std(running_dice)
std_sens = np.std(running_sens)
std_spec = np.std(running_spec)
std_rec = np.std(running_rec)
std_prec = np.std(running_prec)
std_acc = np.std(running_acc)
std_jaccard = np.std(running_jaccard)

print(avg_dice, std_dice)
print(avg_sens, std_sens)
print(avg_spec, std_spec)
print(avg_rec, std_rec)
print(avg_prec, std_prec)
print(avg_acc, std_acc)
print(avg_jaccard, std_jaccard)
