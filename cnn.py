import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2


picture_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(150),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = picture_transformer(self.data[index])
        return data, self.label[index]

    def __len__(self):
        return len(self.data)



class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(512*1*1, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.maxpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):  # x[3, 150, 150]
        x = self.conv1(x)  # x[3, 75, 75]
        x = F.relu(x)

        x = self.conv2(x)   # x[3, 38, 38]
        x = F.relu(x)

        x = self.conv3(x)  # x[3, 19, 19]
        x = F.relu(x)

        x = self.conv4(x)  # x[3, 10, 10]
        x = F.relu(x)

        x = self.conv5(x)  # x[3, 10, 10]
        x = F.relu(x)
        x = self.maxpool(x)  # x[512, 1, 1]

        x=x.view(-1, 512*1*1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

class CNN_new(nn.Module):
    def __init__(self, num_classes):
        super(CNN_new, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(1024*1*1, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.maxpool = nn.AdaptiveAvgPool2d((1, 1))
        self.f = nn.ReLU(inplace=True)

    def forward(self, x):  # x[3, 150, 150]
        x = self.conv1(x)  # x[3, 75, 75]
        x = F.relu(x)

        x = self.conv2(x)   # x[3, 38, 38]
        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)  # x[3, 19, 19]
        x = self.conv5(x)
        x = F.relu(x)

        x = self.conv6(x)  # x[3, 10, 10]
        x = self.conv7(x)
        x = F.relu(x)

        x = self.conv8(x)  # x[3, 10, 10]
        x = F.relu(x)
        x = self.maxpool(x)  # x[512, 1, 1]

        x=x.view(-1, 1024*1*1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sun_loss = 0
    index = 0
    for idx, (data, lable) in enumerate(train_loader):
        data, lable = data.to(device), lable.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, lable).to(device)
        loss.backward()
        optimizer.step()
        sun_loss += loss.item()
        index += 1
        # if idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(epoch, idx, len(data), loss))
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, sun_loss/index))
    return sun_loss/index

def test(model, device, test_loader, len_test_data):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, lable in test_loader:
            data, lable = data.to(device), lable.to(device)
            output = model(data)
            # red = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            pred=output.argmax(dim=1)#batch_size*2->batch_size*1  
            correct += pred.eq(lable.view_as(pred)).sum().item()
    acc=correct/len_test_data
    print("accuracy:{}".format(acc))

def load_data(file_path):
    # 加载数据
    array_of_img = []
    for filename in os.listdir(file_path):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(file_path + "/" + filename)
        array_of_img.append(img)
    return array_of_img

def load_test_data(file_path):
    # 加载数据
    array_of_img = []
    lable = []
    for filename in os.listdir(file_path):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(file_path + "/" + filename)
        array_of_img.append(img)
        if "猫" in filename:
            lable.append(0)
        else:
            lable.append(1)
    return array_of_img, lable


if __name__ == '__main__':
    # 参数设置
    lr = 0.0005
    epoch = 300
    batch_size = 256

    #判断是否使用GPU
    device=torch.device("cuda:7" if torch.cuda.is_available() else "cpu" )
    #加载数据
    cats_data = load_data("/home/jcq/a-gz/Dataset/cats_and_dogs_v2/train/cats")
    print("猫训练数据的大小：", len(cats_data), "第一张图片的大小：", cats_data[0].shape)
    dogs_data = load_data("/home/jcq/a-gz/Dataset/cats_and_dogs_v2/train/dogs")
    print("狗训练数据的大小：", len(dogs_data), "第一张图片的大小：", dogs_data[0].shape)

    #构造标签
    lables = []
    for i in range(len(cats_data)):
        lables.append(0)
    for i in range(len(dogs_data)):
        lables.append(1)

    # 数据构造
    sun_data = dataset(data=cats_data + dogs_data, label=lables)
    train_data = DataLoader(dataset=sun_data, batch_size=batch_size, shuffle=True)

    # 定义model
    # model = CNN(num_classes=2)
    model = CNN_new(num_classes=2)
    model.to(device)
    # 优化器
    # optimizer=optim.Adam(model.parameters(),lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6,
                            weight_decay=1e-3)
    loss = 1
    for e in range(epoch):
        loss_new = train(model=model, train_loader=train_data, optimizer=optimizer, device=device, epoch=e)
        if loss_new == min(loss, loss_new) and e > 200:
            torch.save(model.state_dict(), "/home/jcq/a-gz/me/net/cnn_model.pt")
            loss = loss_new

    #加载数据
    test_data, test_lable = load_test_data("/home/jcq/a-gz/Dataset/cats_and_dogs_v2/test")
    print("测试数据的大小：", len(test_data), "第一张图片的大小：", test_data[0].shape)
    # 数据构造
    te_data = dataset(data=test_data, label=test_lable)
    test_dataset = DataLoader(dataset=te_data, batch_size=batch_size, shuffle=True)
    test(model=model, test_loader=test_dataset, device=device,len_test_data=len(test_data))
