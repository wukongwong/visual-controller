import numpy as np
import os

dataset_dir = "/home/zls/king/visual servo/examples/VTCdataset_linear/seg/"
data_all = []

for file in os.listdir(dataset_dir):
    file_dir = os.path.join(dataset_dir, file)
    for file_ in os.listdir(file_dir):
        # end with *.txt
        if file_.endswith('.txt'):
            file_path = os.path.join(file_dir, file_)
            data = np.loadtxt(file_path, dtype = np.float64)
            data_all.append(data)
            print("append data:", data)

data_all = np.array(data_all)
# calculate the max and min of the data in each row
max_data = np.max(data_all, axis=0)
# calculate the min of the data in each row
min_data = np.min(data_all, axis=0)
np.savetxt("max_data.txt", max_data, fmt='%.20f')
np.savetxt("min_data.txt", min_data, fmt='%.20f')



