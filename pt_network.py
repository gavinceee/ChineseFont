from torch import nn
#init里创建实例，forward使用实例

class Block(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, strides=1, padding='same', alpha=0.2):
        super(Block, self).__init__()
        if kernel_size % 2 == 0:
            padding = kernel_size // 2 -1
        else:
            padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding=padding) #//除于二，取整
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=alpha)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BlockGroup(nn.Module):
    def __init__(self, layers, kernel_size, in_channels, out_channels, strides=1, padding='same', alpha=0.2):
        super(BlockGroup, self).__init__()
        self.conv1 = Block(kernel_size, in_channels, out_channels, strides, padding, alpha)
        self.conv = list()
        self.layers = layers
        for i in range(layers -1):
            setattr(self, f'conv{i+2}', Block(kernel_size, out_channels, out_channels, strides, padding, alpha))
        #for i in range(layers - 1):
        #    self.conv.append(Block(kernel_size, out_channels, out_channels, strides, padding, alpha))

    def forward(self, x):
        x = self.conv1(x)
        #for block in self.conv:
        #    x = block(x)
        for i in range(self.layers -1):
            layer = getattr(self, f"conv{i+2}")
            x = layer(x)
        return x

class RewriteNet(nn.Module): #nn.module 
    def __init__(self, mode):
        super(RewriteNet, self).__init__()
        if mode == 'small':
            print("small model is chosen, shrink number of layers to 2")
            layers = 2
        elif mode == 'big':
            print("big model is chosen, increase number of layers to 4")
            layers = 4
        else:
            layers = 3
        self.conv_64x64 = BlockGroup(layers=2, kernel_size=63, in_channels=1, out_channels=8)
        self.conv_32x32 = BlockGroup(layers=layers, kernel_size=31, in_channels=8, out_channels=32)
        self.conv_16x16 = BlockGroup(layers=layers, kernel_size=15, in_channels=32, out_channels=64)#
        self.conv_7x7   = BlockGroup(layers=layers, kernel_size=7, in_channels=64, out_channels=128)#
        self.conv_3x3_1 = Block(kernel_size=3, in_channels=128, out_channels=128)#
        self.conv_3x3_2 = Block(kernel_size=3, in_channels=128, out_channels=1)#

        self.pooled = nn.MaxPool2d(kernel_size=2, stride=2)#
        self.dropped = nn.Dropout2d(p = 0.9, inplace=False)
        self.y_hat_image = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv_64x64(x)
        x2 = self.conv_32x32(x1)
        x3 = self.conv_16x16(x2)
        x4 = self.conv_7x7(x3)
        x5 = self.conv_3x3_1(x4)
        x6 = self.conv_3x3_2(x5)

        x7 = self.pooled(x6)
        x8 = self.dropped(x7)
        x9 = self.y_hat_image(x8)
        return x9
'''
model = RewriteNet().to(device)

loss_fn = nn.CrossEntropyLoss() # returns how much the two values are off by
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

def Train():
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate (dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y) # pred is the prediction, y is the actual value

        optimizer.zero_grad() # zero_grad 把参数对应的梯度清零
        loss.backward()
        optimizer.step()

    if batch % 100 == 0:
        loss, current = loss.item(), batch * len(X) # loss.item()转化成标量来打印出来
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test (dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): # 设置：没有梯度
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) #to(device) is to change to the cpu/gpu initialed above
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
'''