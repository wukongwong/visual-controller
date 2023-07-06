import time

import torch
import torchvision
import os
import numpy as np
from utils import tf_to_dof, tf_to_quat
from torchvision import transforms
from PIL import Image

model_dir = '/media/zls/My Passport/save_linear_best/VSNet-train/'

class VSNet(object):
    def __init__(self, model, use_gpu= True):

        assert os.path.isfile(model), 'Model does not exists'
        if torch.cuda.is_available() and not use_gpu:
            print('Your computer do have cuda support, please use it')

        if use_gpu:
            assert torch.cuda.is_available(), 'No cuda support'
            self.device = torch.device('cuda')

        self.model = torch.load(model, map_location=self.device)
        self.model = self.model.module
        self.model.eval()
        self.img_transform = transforms.Compose([transforms.ToTensor(),])

    def infer(self,img_a, img_b):
        assert os.path.isfile(img_a), 'The first image does not exist'
        assert os.path.isfile(img_b), 'The second image does not exist'

        img_a = self.img_transform(Image.open(img_a).convert('RGB'))
        start_time = time.time()
        img_b = self.img_transform(Image.open(img_b).convert('RGB'))
        end_time = time.time()

        img_a = img_a.unsqueeze(0)
        img_b = img_b.unsqueeze(0)

        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        output = self.model(img_a, img_b)

        print("time_inner",end_time-start_time)

        output = output.cpu().detach().numpy()

        return np.array(output[0])


def main():
    model = model_dir + 'model.pth'
    img_a = '/home/zls/king/visual servo/examples/VTCdataset_linear/seg/r_0.01n_9/2000.png'
    img_b = '/home/zls/king/visual servo/examples/VTCdataset_linear/seg/r_0.01n_9/2000.png'
    # label_a_path = './six_dof_1cm5deg/label/607.txt'
    # label_b_path = './six_dof_1cm5deg/label/100.txt'
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)


    # tf_a = np.loadtxt(label_a_path, dtype=np.float32)
    # tf_b = np.loadtxt(label_b_path, dtype=np.float32)
    # tf_ba = np.matmul(np.linalg.inv(tf_b), tf_a)
    # label = np.array(tf_to_quat(tf_ba), dtype=np.float32)

    net = VSNet(model)


    start.record()

    # Waits for everything to finish running
    start_time = time.time()
    output = net.infer(img_a, img_b)
    end_time = time.time()
    end.record()
    torch.cuda.synchronize()
    print(output)
    print(start.elapsed_time(end))
    print("time", end_time - start_time)


if __name__ == '__main__':
    main()
