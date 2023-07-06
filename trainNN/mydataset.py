import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

label_0_max = 0.18997402794856910191
label_1_max = 0.19000000000000000222
label_2_max = 0.00000000000000000000
label_3_max = 1.00000000000000000000

label_0_min = -0.18997402794856910191
label_1_min = -0.19000000000000000222
label_2_min = -0.28000000000000002665
label_3_min = 0.00000000000000000000

class VTCDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        # data_info存储所有图片路径和标签路径，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    #obtain the image and label of the dataset
    def __getitem__(self, index):
        # 通过 index 读取样本
        path_img, path_label = self.data_info[index]
        # 注意这里需/要 convert('RGB')
        img = Image.open(path_img).convert('RGB')     # 0~255
        path_goal_img = '/home/zls/king/visual servo/examples/VTCdataset_linear/img_goal.png'
        img_goal = Image.open(path_goal_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)
            img_goal = self.transform(img_goal)# 在这里做transform，转为tensor等

        label = np.loadtxt(path_label)
        # Normalization of label
        label = [(label[0]-label_0_min)/(label_0_max-label_0_min), (label[1]-label_1_min)/(label_1_max-label_1_min),
                 (label[2]-label_2_min)/(label_2_max-label_2_min)]

        p = Path(path_label)
        # get the number part of the label file name
        numb = p.stem
        # string to float
        numb = float(numb)
        label = np.append(label,(1.0-numb/2000.0))
        # # add one more value to label
        # # determine whether the label contains "2000.txt"
        # if path_label.endswith('2000.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1980.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1960.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1940.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1920.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1900.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1880.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1860.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1840.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1820.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1800.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1780.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1760.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1740.txt'):
        #     label = np.append(label, 1)
        # elif path_label.endswith('1720.txt'):
        #     label = np.append(label, 1)
        # else:
        #     label = np.append(label, 0)
        # 返回是样本和标签
        return img, img_goal, label

    # 返回所有样本的数量
    def __len__(self):
        return len(self.data_info)

    #return the address of image and label
    def get_img_info(self,data_dir):
        data_info = list()
        # data_dir 是训练集、验证集或者测试集的路径
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            # dirs [each demonstrations]
            #subdir: such as r_0.11n_65, is one of demonstrations
            for sub_dir in dirs:
                # 文件列表
                demonstration_names = os.listdir(os.path.join(root, sub_dir))

                # 取出 png 结尾的文件
                img_names = list(filter(lambda x: x.endswith('.png'), demonstration_names))
                img_names.sort(key=lambda x:int(x.split('.')[0]))
                # 取出 txt? 结尾的文件
                label_names = list(filter(lambda x: x.endswith('.txt'), demonstration_names))
                label_names.sort(key=lambda x:int(x.split('.')[0]))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    label_name = label_names[i]
                    # 图片的绝对路径
                    path_img = os.path.join(root, sub_dir, img_name)
                    # 标签的绝对路径
                    path_label = os.path.join(root,sub_dir, label_name)

                    # 保存在 data_info 变量中
                    data_info.append((path_img,path_label))
        return data_info