import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os

#Custom dataset class for loading OASIS images
#Further details at https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = os.listdir(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

#--------------
#Data
#Transform for turning images into tensors
transform = transforms.Compose([transforms.ToTensor()])

trainset = CustomDataset('oasisData/keras_png_slices_data/keras_png_slices_data/keras_png_slices_train', transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)

testset = CustomDataset('oasisData/keras_png_slices_data/keras_png_slices_data/keras_png_slices_test', transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

#--------------
#Statistics
#Calculates mean and standard deviation for training and testing sets for normalising data
train_mean = 0
train_std = 0
for i,(images) in enumerate(train_loader):
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    train_mean += images.mean(2).sum(0)
    train_std += images.std(2).sum(0)
    print("Training mean and std: \n", train_mean, train_std)

train_mean /= len(train_loader.dataset)
train_std /= len(train_loader.dataset)

print("Training mean and std: \n", train_mean, train_std)
# mean = 0.1317
# std = 0.1864

test_mean = 0
test_std = 0
for i,(images) in enumerate(test_loader):
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    test_mean += images.mean(2).sum(0)
    test_std += images.std(2).sum(0)
    print("Testing mean and std: \n", test_mean, test_std)

test_mean /= len(test_loader.dataset)
test_std /= len(test_loader.dataset)

print("Testing mean and std: \n", test_mean, test_std)
# mean = 0.1343
# std = 0.1879
