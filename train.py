import torch
import torchvision
import torchvision.transforms as transforms
from dataset import Dataset_Converter
import torch.optim as optim
from torch.autograd import Variable
import argparse
import torch.nn as nn
from net import Discriminator, Generator
from tensorboardX import SummaryWriter
import os
from numpy.random import normal
import numpy as np

#================================options========================================
parser = argparse.ArgumentParser(description='tuning hyparameter')
parser.add_argument('--data_dir', '-d', default='./datasets/')
parser.add_argument('--out_dir', '-o', default='./result/')
parser.add_argument('--epoch', '-e', type=int, default=10000)
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--log_iter', '-l', type=int, default=10)
parser.add_argument('--snapshot_epoch', '-s', type=int, default=100)
parser.add_argument('--display_epoch', '-i', type=int, default=1)
args = parser.parse_args()

data_dir = args.data_dir
out_dir = args.out_dir
epoch = args.epoch
batch_size = args.batch_size
log_iter = args.log_iter
snapshot_epoch = args.snapshot_epoch
display_epoch = args.display_epoch

print('data_dir : {}'.format(data_dir))
print('out_dir : {}'.format(out_dir))
print('epoch : {}'.format(epoch))
print('batch_size : {}'.format(batch_size))
print('log_iter : {}'.format(log_iter))
print('snapshot_epoch : {}'.format(snapshot_epoch))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

if os.path.exists(out_dir) != True:
    os.makedirs(out_dir)
#================================dataset========================================
print("dataset loading")
transforms = transforms.Compose(
    [transforms.ToTensor()])

dataset = Dataset_Converter(root_dir = data_dir, transform = transforms)

trainloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                            shuffle=True, num_workers=2)
print("dataset loaded ({})".format(len(dataset)))

#================================loss===========================================
print("setting model/loss/optimizer")
# x : probability of real from real image
# z : probability of real from fake image

criterion = nn.BCELoss()


def loss_gen(z):
    batch_size = z.size()[0]
    label = torch.ones(batch_size)
    loss = torch.sum(criterion(z[:,0],label.to(device)))
    return loss

# sometimes exchange label to increase the quality of training
def loss_dis_L1(x):
    batch_size = x.size()[0]
    low_possibility = randint(10000)
    label = 0.5
    if low_possibility == 0:
        label = torch.zeros(batch_size)
    else:
        label = torch.ones(batch_size)
    # print('label_L1 : {}'.format(label))
    L1 = torch.sum(criterion(x[:,0],label.to(device)))
    return L1

def loss_dis_L2(z):
    batch_size = z.size()[0]
    low_possibility = randint(10000)
    label = 0.5
    if low_possibility == 0:
        label = torch.ones(batch_size)
    else:
        label = torch.zeros(batch_size)
    # print('label_L2 : {}'.format(label))
    L2 = torch.sum(crit


#================================optimizer======================================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

dis = Discriminator().to(device)
dis.apply(weights_init)
gen = Generator(batch_size).to(device)
gen.apply(weights_init)

dis_opt = optim.Adam(dis.parameters(),lr=0.0002,betas=(0.5,0.999))
gen_opt = optim.Adam(gen.parameters(),lr=0.0002,betas=(0.5,0.999))
print("setted model/loss/optimizer")



#================================train==========================================
print("start training")
iteration_sum = 0
for epo in range(epoch):

    running_loss_dis = 0.0
    running_loss_gen = 0.0
    iterations = 0

    for i, data in enumerate(trainloader, 0):
        iterations = i
        inputs = data

        inputs = Variable(inputs)
        inputs = inputs.to(device)

        # update Discriminator
        #train with inputs
        dis_opt.zero_grad()
        x = dis(inputs,epo)
        L1 = loss_dis_L1(x)
        L1.backward()

        #train with fake
        z = gen(epo)
        z_out_1 = dis(z.detach(),epo)
        L2 = loss_dis_L2(z_out_1)
        L2.backward()
        dis_opt.step()

        writer.add_scalar('dis_loss', L1+L2, iteration_sum + i)

        # updata Generator
        gen_opt.zero_grad()
        z_out_2 = dis(z,epo)
        gen_loss = loss_gen(z_out_2)
        gen_loss.backward()
        gen_opt.step()

        writer.add_scalar('gen_loss', gen_loss, iteration_sum + i)
        writer.add_scalars('loss',{'dis' : L1+L2,
                                    'gen' : gen_loss},iteration_sum + i)

        #log
        running_loss_dis += L1.item() + L2.item()
        running_loss_gen += gen_loss.item()
        if (iteration_sum + i) % log_iter == 0:
            print('epoch : {0}, iter : {1}, dis_loss : {2}, gen_loss : {3}'.format(epo, iteration_sum + i,
                            round(running_loss_dis / log_iter, 3), round(running_loss_gen / log_iter, 3)))
            running_loss_dis = 0.0
            running_loss_gen = 0.0
    iteration_sum += iterations

    if epo % snapshot_epoch == 0 and epo != 0:
        torch.save(dis.state_dict(), out_dir + '/dis_epoch_{}.pth'.format(epo))
        torch.save(gen.state_dict(), out_dir + '/gen_epoch_{}.pth'.format(epo))

    if epo % display_epoch == 0 and epo != 0:
        # 4 * 4 display image
        create_img = torch.zeros(3,128*4,128*4)
        create_tensor = gen(epo)
        for j in range(4):
            for k in range(4):
                create_img[:,j*128:(j+1)*128,k*128:(k+1)*128] = create_tensor.detach()[j*4+k]
        torchvision.utils.save_image(create_img,out_dir + '/create_epoch_{}.jpg'.format(epo))



        writer.add_image('Create_Image', create_img, epo)


writer.export_scalars_to_json("./all_scalrs.json")
writer.close()
