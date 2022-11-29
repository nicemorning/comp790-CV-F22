import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

# PyTorch FORWARD-MODE AUTOMATIC DIFFERENTIATION
# https://pytorch.org/tutorials/intermediate/forward_ad_usage.html
import torch.autograd.forward_ad as fwAD
from torch.nn.utils._stateless import functional_call

from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

        
# LeNet for CIFAR-10 Input 3x32x32
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# LeNet for MNIST  Input 1x28x28
class LeNetMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# MLP for MNIST  784->1024->128->10
class MLPMnist1(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(784, 1024),
      nn.ReLU(),
      nn.Linear(1024, 128),
      nn.ReLU(),
      nn.Linear(128, 10)
    )

  def forward(self, x):
    '''Forward pass'''
    # flatten image input
    x = x.view(-1, 28 * 28)
    return self.layers(x)


# MLP for MNIST  784->16->10
class MLPMnist2(nn.Module):

  def __init__(self, h=16):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(784, h),
      nn.ReLU(),
      nn.Linear(h, 10),
    )

  def forward(self, x):
    # flatten image input
    x = x.view(-1, 28 * 28)
    x = self.layers(x)
    return x


# CIFAR-10 Dataset
def dataset_cifar10(num_workers=0):
    print('==> Preparing CIFAR-10 dataset')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f'TrainBatch:{len(trainloader)} TestBatch:{len(testloader)}')
    return trainloader, testloader


# Training
def train_fgd(epoch, lr=0.001):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    dual_params = {}
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        tangents = {name: torch.randn_like(p) for name, p in params.items()}
        
        with fwAD.dual_level():
            for name, p in model.named_parameters():
                dual_params[name] = fwAD.make_dual(p, tangents[name])
        
            outputs = functional_call(model, dual_params, inputs)
            loss_dual = criterion(outputs, targets)
            jvp = fwAD.unpack_dual(loss_dual).tangent
            loss = fwAD.unpack_dual(loss_dual).primal
        
        #print(jvp)
        #print(loss)
        
        with torch.no_grad():  
            for name, p in model.named_parameters():
                p_update = p - lr * jvp * tangents[name]
                p.copy_(p_update)
        
          
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        #print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    t1 = time.time()
    print(f'TrainFGD Iter:{epoch} Loss:{train_loss/(batch_idx+1):.3f} Acc:{100.*correct/total:.3f}% Time:{(t1-t0):.3f}') 
    
    

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    t1 = time.time()        

    #print(epoch, 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(f'Test     Iter:{epoch} Loss:{test_loss/(batch_idx+1):.3f} Acc:{100.*correct/total:.3f}% Time:{(t1-t0):.3f}') 

def train_sgd(epoch, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
          
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    t1 = time.time()
    print(f'TrainSGD Iter:{epoch} Loss:{train_loss/(batch_idx+1):.3f} Acc:{100.*correct/total:.3f}% Time:{(t1-t0):.3f}') 
    

def sgd_init(lr=0.1):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return optimizer

def fgd_init():
    params = {name: p for name, p in model.named_parameters() if p.requires_grad==True}
    return params

FGD = True    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device:{device}')


# Model
print('==> Building model..')

model = LeNet()
model = model.to(device)
print(model)

# Loss function
criterion = nn.CrossEntropyLoss()

if FGD == True: 
    params = fgd_init()
else:     
    optimizer = sgd_init(lr=0.01)


trainloader, testloader = dataset_cifar10()

for epoch in range(20):
    if FGD == True:
        train_fgd(epoch)
    else:
        train_sgd(epoch, optimizer)
    
    test(epoch)
    #scheduler.step()
   
