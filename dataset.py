from torch.utils.data import DataLoader, Dataset
import numpy as np
from config import CLASSES
import matplotlib.pyplot as plt
import torch
import os
import cv2 as cv

class DataQuickDraw(Dataset):
    def __init__(self, root_path, total_images_per_class = 10000,is_train= True, ratio= 0.8):
        self.root_path = root_path
        self.num_classes = len(CLASSES)
        if is_train :
            self.offset = 0
            self.num_images_per_class = int(total_images_per_class *ratio)
        else :
            self.offset = int(total_images_per_class *ratio)
            self.num_images_per_class = int(round(total_images_per_class  * (1-ratio)))
        self.num_samples = self.num_images_per_class * self.num_classes
    def __len__(self):
        return self.num_samples
    def __getitem__(self, item):
        class_index = item//self.num_images_per_class
        file_ = os.path.join(self.root_path, "full_numpy_bitmap_{}.npy".format(CLASSES[int(item/self.num_images_per_class)]))
        data = np.load(file_)
        image = data[self.offset + (item % self.num_images_per_class)].astype(np.float32)
        image /= 255
        return image.reshape((1, 28, 28)), torch.tensor(class_index, dtype=torch.long)

if __name__ == '__main__':
    root_path = "quick_draw"
    # root_path = "/kaggle/input"
    data_train = DataQuickDraw(root_path, is_train=True)
    # data_val = DataQuickDraw(root_path, is_train=False)
    # train_loader = DataLoader(data_train,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=4,
    #     drop_last=True
    #     )
#     val_loader = DataLoader(data_val,
#     batch_size=8,
#     shuffle=False,
#     num_workers= 4
#     )
#     for image, label in val_loader :
#         print(image.shape, label.shape)
    # print(data_val.__len__())
    image, label = data_train[68989]
    image_cv = (image.squeeze() * 255).astype(np.uint8)
    cv.imwrite("eye_3.jpg", image_cv)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis()
    plt.show()




