# -*- coding: utf-8 -*-
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
import tensorboard_logger as tb
import mydataset
from model import VSNet
from utils import check_dir

# model_name = 'VSNet-train'#20230627
model_name = 'online_version'
model_pretrained = None
#reaching and gripper command, output is [x,y,z,gripper command]
num_classes = 4


root_dir = "/home/zls/king/visual servo/examples/VTCdataset"
save_root_dir = '/home/zls/king/visual servo/visualtocontrol/save'

# num_epochs = 2#origin 20230627
num_epochs = 10
BATCH_SIZE = 256
aug_factor = 0.25
num_workers = 8

# learning_rate = 1e-4
learning_rate = 1e-5

# milestones = [10, 20, 30]
# milestones = [int(num_epochs * 0.4), int(num_epochs * 0.6),int(num_epochs * 0.8)]
# milestones = [int(4), int(6),int(8)]
milestones = [int(4), int(6),int(8)]


# milestones = list(range(num_epochs))
gamma = 0.5
momentum = 0.9
# gamma = 0.3
limits = None
weights = [0.99, 0.01]

mode = ('train', 'train')  # train train set
# mode = ('eval', 'train')  # eval train set
# mode = ('eval', 'dev')  # eval dev set
# mode = ('eval', 'test')  # eval test set

train_dir = "/home/zls/king/visual servo/examples/VTCdataset_linear/train_dir"
valid_dir = '/home/zls/king/visual servo/examples/VTCdataset_linear/validation_dir'
test_dir = '/home/zls/king/visual servo/examples/VTCdataset_linear/test_dir'



# 构建MyDataset实例
train_data = mydataset.VTCDataset(data_dir=train_dir,transform= transforms.ToTensor())
validation_data = mydataset.VTCDataset(data_dir=valid_dir,transform= transforms.ToTensor())
# test_data = mydataset.VTCDataset(data_dir=train_dir,transform= transforms.ToTensor())

# valid_data = mydataset.VTCDataset(data_dir=valid_dir)

# 构建DataLoder
# 其中训练集设置 shuffle=True，表示每个 Epoch 都打乱样本
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

def mode_train():
    check_dir(save_root_dir + '/' + model_name)

    device = torch.device('cuda')

    if model_pretrained:
        print('Loading pretrained model from {}'.format(save_root_dir + '/' + model_pretrained + '/model.pth'))
        model = torch.load(save_root_dir + '/' + model_pretrained + '/model.pth', map_location=device)
    else:
        model = VSNet(num_classes=num_classes)
        model = nn.DataParallel(model, device_ids=[0, 1, 2])

    # criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    model.to(device)

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


    tb.configure(save_root_dir + '/' + model_name)

    start_time = time.time()

    tb_count = 0
    tb_count_val = 0

    for epoch in range(num_epochs):

        # scheduler.step()

        # Training
        model.train()
        running_loss = 0.0
        running_loss_val = 0.0

        for i, sample in enumerate(train_loader, 0):
            if i == 1 and epoch == 0:
                start_time = time.time()
            img_a, img_b, label = sample
            # clear the gradient
            optimizer.zero_grad()

            img_a = img_a.to(device)
            img_b = img_b.to(device)
            label = label.to(device)
            # make label datatype same as the output
            label = label.float()
            output = model(img_a, img_b)

            # reaching loss
            criterion = nn.MSELoss()
            # binary cross entropy
            # criterion_ = nn.BCELoss()
            loss = torch.sqrt(criterion(output, label))



            # loss = combined_loss_quat(output, label, weights=weights)
            # print(output.dtype)
            # print(label.dtype)
            # print(loss.dtype)
            loss.backward()

            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * output.shape[0]

            output = output.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            est_time = (time.time() - start_time) / (epoch * len(train_loader) + i + 1) * (
                    num_epochs * len(train_loader))
            est_time = str(datetime.timedelta(seconds=est_time))
            print(
                '[TRAIN][{}][EST:{}] Epoch {}, Batch {}, Loss = {:0.7f}'.format(
                    time.time() - start_time, est_time, epoch + 1, i + 1,
                    loss.item()))

            tb.log_value(name='Loss_train', value=loss.item(), step=tb_count)
            tb_count += 1

        # Validation
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(validation_loader, 0):
                img_a, img_b, label = sample

                img_a = img_a.to(device)
                img_b = img_b.to(device)
                label = label.to(device)
                # make label datatype same as the output
                label = label.float()
                output = model(img_a, img_b)

                criterion = nn.MSELoss()
                loss_validation = torch.sqrt(criterion(output, label))

                running_loss_val += loss_validation.item() * output.shape[0]

                print(
                    '[VALIDATION] Epoch {}, Batch {}, Loss = {:0.7f}'.format(
                        epoch + 1, i + 1,loss_validation.item()))
                tb.log_value(name='Loss_val', value=loss_validation.item(), step=tb_count_val)
                tb_count_val += 1



        average_loss = running_loss / len(train_loader)
        average_loss_val = running_loss_val / len(validation_loader)

        print(
            '[SUMMARY][{}] Summary: Epoch {}, loss = {:0.7f}\n'.format(
                time.time() - start_time, epoch + 1, average_loss))

        tb.log_value(name='Loss_epoch_train', value=average_loss, step=epoch)
        tb.log_value(name='Loss_epoch_validation', value=average_loss_val, step=epoch)



        # torch.save(model, save_root_dir + '/' + model_name + '/model.pth')
        torch.save(model, save_root_dir + '/' + model_name + '/' + str(epoch) +'model.pth')

        print('Model saved at {}/{}/model.pth'.format(save_root_dir, model_name))

if __name__ == '__main__':
  mode_train()
