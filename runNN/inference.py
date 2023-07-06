import torch
import torchvision
import os
import numpy as np
# from utils import tf_to_dof, tf_to_quat
from torchvision import transforms
from PIL import Image

model_dir = '/home/zls/king/visual servo/visualtocontrol/save/VSNet-train/'

class VSNet(object):
    def __init__(self, model, use_gpu= True):

        assert os.path.isfile(model), 'Model does not exists'
        if torch.cuda.is_available() and not use_gpu:
            print('Your computer do have cuda support, please use it')

        if use_gpu:
            assert torch.cuda.is_available(), 'No cuda support'
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = torch.load(model, map_location=self.device)
        self.model = self.model.module
        self.model.eval()
        self.img_transform = transforms.Compose([transforms.ToTensor(),])

    def infer(self,img_a, img_b):
        # assert os.path.isfile(img_a), 'The first image does not exist'
        assert os.path.isfile(img_b), 'The second image does not exist'

        img_a = self.img_transform(img_a.convert('RGB'))
        img_b = self.img_transform(Image.open(img_b).convert('RGB'))
        img_a = img_a.unsqueeze(0)
        img_b = img_b.unsqueeze(0)

        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        output = self.model(img_a, img_b)
        output = output.cpu().detach().numpy()

        return np.array(output[0])

class VSNet_realtime(object):
    def __init__(self, model, use_gpu=True):

        assert os.path.isfile(model), 'Model does not exists'
        if torch.cuda.is_available() and not use_gpu:
            print('Your computer do have cuda support, please use it')

        if use_gpu:
            assert torch.cuda.is_available(), 'No cuda support'
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = torch.load(model, map_location=self.device)
        self.model = self.model.module
        self.model.eval()
        self.img_transform = transforms.Compose([transforms.ToTensor(), ])

    def infer(self, img_a, img_b):
        # assert os.path.isfile(img_a), 'The first image does not exist'
        assert os.path.isfile(img_b), 'The second image does not exist'

        img_a = self.img_transform(img_a.convert('RGB'))
        img_b = self.img_transform(Image.open(img_b).convert('RGB'))
        img_a = img_a.unsqueeze(0)
        img_b = img_b.unsqueeze(0)

        img_a = img_a.to(self.device)
        img_b = img_b.to(self.device)
        output = self.model(img_a, img_b)
        output = output.cpu().detach().numpy()

        return np.array(output[0])


def main():
    model = model_dir + 'model.pth'
    img_a = '/home/zls/king/visual servo/examples/VTCdataset/seg/r_0.1n_19/0.png'
    img_b = '/home/zls/king/visual servo/examples/VTCdataset/img_goal.png'
    # label_a_path = './six_dof_1cm5deg/label/607.txt'
    # label_b_path = './six_dof_1cm5deg/label/100.txt'
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    net = VSNet(model)

    start.record()

    # Waits for everything to finish running
    output = net.infer(img_a, img_b)
    end.record()
    torch.cuda.synchronize()
    print(output)
    print(start.elapsed_time(end))

if __name__ == '__main__':
    main()
