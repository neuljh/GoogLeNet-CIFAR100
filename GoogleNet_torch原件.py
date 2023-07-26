import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torchvision
import numpy as np



class Inception(nn.Module):
    def __init__(self, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_1_x, kernel_size=1),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_3_in, kernel_size=1),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_3_in, kernel_3_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_5_in, kernel_size=1),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_in, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_x, kernel_5_x, kernel_size=3, padding=1),#2 3*3=>1 5*5
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self,in_channels):
        super(GoogLeNet, self).__init__()
        self.pre_layers1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.pre_layers2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.drop_out = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, 100)


    def forward(self, x):
        x = self.pre_layers1(x)
        x = self.pre_layers2(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = self.drop_out(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

import torch
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def load_data():
    # 定义数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载数据集
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8,pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8,pin_memory=True)

    # 加载验证集
    validset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    # 定义类别名称
    classes = tuple(trainset.classes)
    return trainloader,testloader,validloader,classes



def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GoogLeNet(3).to(device)
    return net,device

def get_config(net):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    return criterion,optimizer

def train_model(net, criterion, optimizer, trainloader, valloader, device,epochs=10):
    # Train the model
    net.to(device)
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    train_loss_iter_list = []
    train_acc_iter_list = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("Epoch[{}/{}]:".format(epoch + 1, epochs))
        running_train_loss = 0.0
        running_train_acc = 0.0
        running_train_iter_loss = 0.0
        running_train_iter_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_train_iter_loss += loss.item()
            running_train_iter_acc += acc.item()
            running_train_loss += loss.item()
            running_train_acc += acc.item()

            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] train_loss: %.3f, train_accuracy: %.3f' %
                      (epoch + 1, i + 1, running_train_iter_loss / 100, running_train_iter_acc / 100))
                train_loss_iter_list.append(running_train_iter_loss / 100)
                train_acc_iter_list.append(running_train_iter_acc / 100)
                running_train_iter_loss = 0.0
                running_train_iter_acc = 0.0

        train_loss_list.append(running_train_loss / len(trainloader))
        train_acc_list.append(running_train_acc / len(trainloader))
        print('Epoch[ %d / %d ] : train_loss: %.3f, train_accuracy: %.3f' % (epoch + 1, epochs, running_train_loss / len(trainloader), running_train_acc / len(trainloader)))


        # Evaluate the model on the validation set
        running_val_loss = 0.0
        running_val_acc = 0.0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                val_loss = criterion(outputs, labels)
                val_acc = accuracy(outputs, labels)
                running_val_loss += val_loss.item()
                running_val_acc += val_acc.item()
        val_loss_list.append(running_val_loss / len(valloader))
        val_acc_list.append(running_val_acc / len(valloader))
        print('Epoch[ %d / %d ] : val_loss: %.3f, val_accuracy: %.3f' %
              (epoch+1, epochs, running_val_loss / len(valloader), running_val_acc / len(valloader)))

    print('Finished Training')

    # Plot the loss and accuracy curves
    plt.plot(train_loss_iter_list, label='train')
    # plt.plot(val_loss_list, label='val')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(train_acc_iter_list, label='train')
    # plt.plot(val_acc_list, label='val')
    plt.title('Accuracy Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(train_acc_list, label='train')
    plt.plot(val_acc_list, label='val')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def test_model(net,testloader,criterion,device):
    # Test the model
    correct = 0
    total = 0
    test_loss = 0.0
    predictions=[]
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels).item()
            predictions.append(predicted.cpu().numpy())

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print('Test loss: %.3f' % (test_loss / len(testloader)))

    predictions = np.concatenate(predictions)
    class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                   'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                   'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                   'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                   'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                   'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
                   'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
                   'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                   'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
                   'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
                   'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    # 随机选择 5*5 张图像进行可视化
    images, labels = iter(testloader).__next__()
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        # 显示图像
        ax.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))

        # 判断预测结果是否与真实标签相同
        predicted_label = class_names[predicted[i]]
        true_label = class_names[labels[i]]
        if predicted_label == true_label:
            label_color = 'green'
            label_text = 'Correct'
        else:
            label_color = 'red'
            label_text = 'Wrong'

        # 设置图像标题为分类名称和结果标签
        ax.set_title(f"Predicted: {predicted_label} \n results: {label_text} \n True: {true_label}", color=label_color)

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    trainloader, testloader,validloader,classes=load_data()
    net,device = get_model()
    print('current device:  ', device)
    criterion,optimizer=get_config(net)
    train_model(net, criterion, optimizer, trainloader, validloader, device)
    test_model(net, testloader, criterion, device)




