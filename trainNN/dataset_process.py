import random,shutil,os

root_dir = "/home/zls/king/visual servo/examples/VTCdataset_linear/"
train_dir='/home/zls/king/visual servo/examples/VTCdataset_linear/train_dir'
validation_dir='/home/zls/king/visual servo/examples/VTCdataset_linear/validation_dir'
test_dir='/home/zls/king/visual servo/examples/VTCdataset_linear/test_dir'

#divide dataset to train set, validation set, test set
def dataset_division(data_dir):

    folders = os.listdir(data_dir+'/seg')
    total_num = len(folders)
    train_ratio = 0.8
    validation_ratio = 0.5

    train_folders_num = int(total_num*train_ratio)

    train = random.sample(folders,train_folders_num)

    folders_without_train = set(folders)-set(train)
    num_ = len(folders_without_train)
    validation_folders_num =int(num_*validation_ratio)
    validation = random.sample(folders_without_train,validation_folders_num)


    test = set(folders_without_train)-set(validation)

    for sample in train:
        for root,dirs,files in os.walk(data_dir+'seg'+'/'+sample):
            for file in files:
                src_file =os.path.join(root,file)
                if not os.path.exists(train_dir+'/'+sample):
                    os.mkdir(train_dir+'/'+sample)
                shutil.copy(src_file,train_dir+'/'+sample)

        print("train data creating...")

    for sample in validation:
        for root,dirs,files in os.walk(data_dir+'seg'+'/'+sample):
            for file in files:
                src_file =os.path.join(root,file)
                if not os.path.exists(validation_dir+'/'+sample):
                    os.mkdir(validation_dir+'/'+sample)
                shutil.copy(src_file,validation_dir+'/'+sample)

        print("validation data creating...")

    for sample in test:
        for root,dirs,files in os.walk(data_dir+'seg'+'/'+sample):
            for file in files:
                src_file =os.path.join(root,file)
                if not os.path.exists(test_dir+'/'+sample):
                    os.mkdir(test_dir+'/'+sample)
                shutil.copy(src_file,test_dir+'/'+sample)

        print("test data creating...")

    pass

if __name__ == '__main__':
    dataset_division(root_dir)