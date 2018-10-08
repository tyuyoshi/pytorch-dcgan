import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        super(Generator,self).__init__()
        # input : 100
        # output : (3,128,128)
        self.linear1 = nn.Linear(100,512*4*4,bias=False)
        self.conv1 = nn.ConvTranspose2d(512,256,4,stride=2,padding=1,bias=False)
        self.conv2 = nn.ConvTranspose2d(256,128,4,stride=2,padding=1,bias=False)
        self.conv3 = nn.ConvTranspose2d(128,64,4,stride=2,padding=1,bias=False)
        self.conv4 = nn.ConvTranspose2d(64,32,4,stride=2,padding=1,bias=False)
        self.conv5 = nn.ConvTranspose2d(32,16,4,stride=2,padding=1,bias=False)
        self.conv6 = nn.ConvTranspose2d(16,3,3,stride=1,padding=1,bias=False)
        self.linear1_bn = nn.BatchNorm1d(512*4*4)
        self.conv1_bn = nn.BatchNorm2d(256)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv5_bn = nn.BatchNorm2d(16)

    def forward(self, epochs):
        # z = 100 dimentional vector
        z = torch.randn(self.batch_size, 100)
        z = z.to(device)
        z = F.relu(self.linear1_bn(self.linear1(z)))
        z = z.view(-1,512,4,4)
        z = F.relu(self.conv1_bn(self.conv1(z)))
        z = F.relu(self.conv2_bn(self.conv2(z)))
        z = F.relu(self.conv3_bn(self.conv3(z)))
        z = F.relu(self.conv4_bn(self.conv4(z)))
        z = F.relu(self.conv5_bn(self.conv5(z)))
        z = F.sigmoid(self.conv6(z))
        return z

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()
        # input : (3,128,128)
        # output : probability of real

        self.conv1 = nn.Conv2d(3,8,4,stride=2,padding=1)
        self.conv1_1 = nn.Conv2d(8,32,3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,4,stride=2,padding=1)
        self.conv2_1 = nn.Conv2d(32,64,3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(64,64,4,stride=2,padding=1)
        self.conv3_1 = nn.Conv2d(64,256,3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(256,256,4,stride=2,padding=1)
        self.conv4_1 = nn.Conv2d(256,512,4,stride=2,padding=1)
        self.linear1 = nn.Linear(512*4*4,1)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv5_bn = nn.BatchNorm2d(8)

    def forward(self,x,epochs):
        x = self.addGaussianNoise(x,epochs)
        # x = (3,128,128)
        x = F.leaky_relu(self.addGaussianNoise(self.conv1_bn(self.conv1(x)),epochs))
        x = F.leaky_relu(self.addGaussianNoise(self.conv1_1_bn(self.conv1_1(x)),epochs))
        x = F.leaky_relu(self.addGaussianNoise(self.conv2_bn(self.conv2(x)),epochs))
        x = F.leaky_relu(self.addGaussianNoise(self.conv2_1_bn(self.conv2_1(x)),epochs))
        x = F.leaky_relu(self.addGaussianNoise(self.conv3_bn(self.conv3(x)),epochs))
        x = F.leaky_relu(self.addGaussianNoise(self.conv3_1_bn(self.conv3_1(x)),epochs))
        x = F.leaky_relu(self.addGaussianNoise(self.conv4_bn(self.conv4(x)),epochs))
        x = F.leaky_relu(self.addGaussianNoise(self.conv4_1_bn(self.conv4_1(x)),epochs))
        x = x.view(-1,512*4*4)

        x = F.sigmoid(self.linear1(x))
        # x : probability
        return x

    def addGaussianNoise(self,x,epoch):
        batch,ch,row,col = x.size()
        means = torch.zeros(batch,ch,row,col)
        std = 0.01 / (epoch + 1)
        stds = torch.ones(batch,ch,row,col) * std
        gauss = torch.normal(means,stds)
        gauss = gauss.to(device)
        x = x + gauss
        return x
