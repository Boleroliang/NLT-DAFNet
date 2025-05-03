import torch
import torch.nn as nn
from torch.nn import AdaptiveAvgPool2d,Conv2d,BatchNorm2d,ReLU,MaxPool2d

class TreeNode(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TreeNode, self).__init__()
        self.fc_left = nn.Linear(input_dim, output_dim)
        self.fc_right = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        left_output = self.fc_left(x)
        right_output = self.fc_right(x)
        return left_output, right_output

class BinaryTreeLayer(nn.Module):
    def __init__(self, input_dim, output_dim, depth):
        super(BinaryTreeLayer, self).__init__()
        self.depth = depth
        self.root = TreeNode(input_dim, output_dim)
        self.build_tree(self.root, depth - 1, output_dim)
        
    def build_tree(self, node, depth, output_dim):
        if depth > 0:
            node.left = TreeNode(output_dim, output_dim)
            node.right = TreeNode(output_dim, output_dim)
            self.build_tree(node.left, depth - 1, output_dim)
            self.build_tree(node.right, depth - 1, output_dim)
        else:
            node.left = None
            node.right = None

    def forward(self, x):
        return self._forward(x, self.root)
    
    def _forward(self, x, node):
        if node.left is None and node.right is None:
            return node.fc_left(x) + node.fc_right(x)
        left_output, right_output = node(x)
        left_output = self._forward(left_output, node.left)
        right_output = self._forward(right_output, node.right)
        return left_output + right_output


# downsample
class downsample(nn.Module):
    def __init__(self, input_channel, output_channel,kernel_size=3,strides=2,padding=1):
        super(downsample, self).__init__()
        self.conv = Conv2d(in_channels=input_channel, out_channels= output_channel, kernel_size=kernel_size, stride=strides, padding=padding)
        self.bn   = BatchNorm2d(output_channel)
        self.relu = ReLU(inplace=False)

    def forward(self, x):        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

# basic block
class BasicBlock(nn.Module):
    def __init__(self,input_channel,out_channel,kernel_size=3,strides=1,padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_channels=input_channel, out_channels= out_channel, kernel_size=kernel_size, stride=strides, padding=padding)
        self.bn    = BatchNorm2d(out_channel)
        self.relu  = ReLU(inplace=False)

        self.conv2 = Conv2d(in_channels=out_channel, out_channels= out_channel, kernel_size=kernel_size, stride=strides, padding=padding)
        self.bn2   = BatchNorm2d(out_channel)
        # self.relu2 = ReLU(inplace=False)
        self.downsample = downsample(input_channel, out_channel, kernel_size=kernel_size, strides=strides)
        self.relu2 = ReLU(inplace=False)

    def forward(self, x):
        identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.relu2(x)

        x = x + identity
        x = self.relu2(x)

        return x

# refine feture module
class RefineFeture(nn.Module):
    def __init__(self,input_channel,out_channel,dliation_num,kernel_size=1,strides=1,dliation=1):
        super(RefineFeture, self).__init__()
        self.daconv1 = Conv2d(in_channels=input_channel, out_channels= out_channel, kernel_size=kernel_size, stride=strides, padding=dliation*(kernel_size//2), dilation=dliation*dliation_num[0])
        self.bn1     = BatchNorm2d(out_channel)
        # self.relu1   = ReLU(inplace=False)

        self.daconv2 = Conv2d(in_channels=input_channel, out_channels= out_channel, kernel_size=kernel_size, stride=strides, padding=dliation*(kernel_size//2), dilation=dliation*dliation_num[1])
        self.bn2     = BatchNorm2d(out_channel)
        # self.relu2   = ReLU(inplace=False)

        self.daconv3 = Conv2d(in_channels=input_channel, out_channels= out_channel, kernel_size=kernel_size, stride=strides, padding=dliation*(kernel_size//2), dilation=dliation*dliation_num[2])
        self.bn3     = BatchNorm2d(out_channel)
        # self.relu3   = ReLU(inplace=False)

        self.daconv4 = Conv2d(in_channels=input_channel, out_channels= out_channel, kernel_size=kernel_size, stride=strides, padding=dliation*(kernel_size//2), dilation=dliation*dliation_num[3])
        self.bn4     = BatchNorm2d(out_channel)
        # self.relu4   = ReLU(inplace=False)

        self.relu    = ReLU(inplace=False)

    def forward(self, x):
        x1 = self.daconv1(x)
        x1 = self.bn1(x1)
        # x1 = self.relu1(x1)

        x2 = self.daconv2(x)
        x2 = self.bn2(x2)
        # x2 = self.relu2(x2)

        x3 = self.daconv3(x)
        x3 = self.bn3(x3)
        # x3 = self.relu3(x3)

        x4 = self.daconv4(x)
        x4 = self.bn4(x4)
        # x4 = self.relu4(x4)
        
        assert x1.shape == x2.shape == x3.shape == x4.shape

        x = torch.sum(torch.stack([x1, x2, x3, x4]), dim=0)

        x = self.relu(x)

        return x

# multi scale convolution
class MSC(nn.Module):
    def __init__(self,input_channel,out_channel,strides=1):
        super(MSC, self).__init__()

        #  mutli scale
        self.conv1   = Conv2d(in_channels=input_channel, out_channels= out_channel, kernel_size=(1,1), stride=strides, padding='same')
        self.bn1     = BatchNorm2d(out_channel)

        self.conv2_1 = Conv2d(in_channels=input_channel, out_channels= out_channel, kernel_size=(1,3), stride=strides, padding='same')
        self.bn2_1   = BatchNorm2d(out_channel)
        self.relu2_1 = ReLU(inplace=False)
        self.conv2_2 = Conv2d(in_channels=out_channel,   out_channels= out_channel, kernel_size=(3,1), stride=strides, padding='same')
        self.bn2_2   = BatchNorm2d(out_channel)

        self.conv3_1 = Conv2d(in_channels=input_channel, out_channels= out_channel, kernel_size=(5,1), stride=strides, padding='same')
        self.bn3_1   = BatchNorm2d(out_channel)
        self.relu3_1 = ReLU(inplace=False)
        self.conv3_2 = Conv2d(in_channels=out_channel,   out_channels= out_channel, kernel_size=(1,5), stride=strides, padding='same')
        self.bn3_2   = BatchNorm2d(out_channel)

        self.conv4_1 = Conv2d(in_channels=input_channel, out_channels= out_channel, kernel_size=(1,7), stride=strides, padding='same')
        self.bn4_1   = BatchNorm2d(out_channel)
        self.relu4_1 = ReLU(inplace=False)
        self.conv4_2 = Conv2d(in_channels=out_channel,   out_channels= out_channel, kernel_size=(7,1), stride=strides, padding='same')
        self.bn4_2   = BatchNorm2d(out_channel)

        self.relu    = ReLU(inplace=False)
    
    def forward(self, x):
        #  mutli scale
        x1 = self.conv1(x)
        x1 = self.bn1(x1)

        x2_1 = self.conv2_1(x)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu2_1(x2_1)
        x2_2 = self.conv2_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        x3_1 = self.conv3_1(x)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu3_1(x3_1)
        x3_2 = self.conv3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        x4_1 = self.conv4_1(x)
        x4_1 = self.bn4_1(x4_1)
        x4_1 = self.relu4_1(x4_1)
        x4_2 = self.conv4_2(x4_1)
        x4_2 = self.bn4_2(x4_2)

        assert x1.shape == x2_2.shape == x3_2.shape == x4_2.shape
    
        x = torch.sum(torch.stack([x1, x2_2, x3_2, x4_2]), dim=0)

        x = self.relu(x)

        return x
    
# multi scale channel and spatial mechanisms
class MSCSM(nn.Module):
    def __init__(self,input_channel,out_channel,strides=1):
        super(MSCSM, self).__init__()

        # channel
        self.gap  = AdaptiveAvgPool2d(1)
        self.MSC1 = MSC(input_channel,out_channel,strides=strides)
        self.conv = Conv2d(in_channels=out_channel, out_channels= out_channel, kernel_size=1, stride=strides)

        # spatial
        self.MSC2 = MSC(input_channel,out_channel,strides=strides)
        self.conv = Conv2d(in_channels=out_channel, out_channels= out_channel, kernel_size=1, stride=strides)
    
    def forward(self, x):
        #  channel
        x1 = self.gap(x)
        x1 = self.MSC1(x1)
        x1 = self.conv(x1)
        x1 = x*x1
        
        # spatial
        x2 = self.MSC2(x)
        x2 = self.conv(x2)
        x2 = x*x2

        assert x1.shape == x2.shape

        x = torch.sum(torch.stack([x1, x2]), dim=0)

        return x

# dual fine channel and spatial module
class DFCSM(nn.Module):
    def __init__(self,input_channel,out_channel,dm,strides=1):
        super(DFCSM, self).__init__()

        self.rf   = RefineFeture(input_channel,out_channel,dliation_num=dm, strides=strides)
        self.cs   = MSCSM(out_channel,out_channel,strides=strides)
        self.bb   = BasicBlock(out_channel,out_channel,strides=strides)
        self.down = downsample(input_channel,out_channel,strides=strides)
    

    def forward(self, x):
        x1 = self.rf(x)
        x2 = self.cs(x1)
        x3 = self.bb(x2)
        x4 = self.down(x)

        assert x3.shape == x4.shape

        x = torch.sum(torch.stack([x3, x4]), dim=0)

        return x

# residual fine-grained attention convolution aggregation model
class RFACA_Net(nn.Module):
    def __init__(self,num_classes,depth,calibration_iter=10,input_channel=3,out_channel=64,dm=[1,4,8,12],strides=1):
        super(RFACA_Net, self).__init__()
        self.bb    = BasicBlock(input_channel,out_channel,strides=strides)
        self.max   = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dfcsm1 = DFCSM(out_channel,out_channel,dm,strides=strides)
        self.max1   = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dfcsm2 = DFCSM(out_channel,out_channel*2,dm,strides=strides)
        self.max2   = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dfcsm3 = DFCSM(out_channel*2,out_channel*4,dm,strides=strides)
        self.max3   = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dfcsm4 = DFCSM(out_channel*4,out_channel*8,dm,strides=strides)
        self.max4   = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.gap    = AdaptiveAvgPool2d(1)
        # self.fc     = nn.Linear(out_channel*8, num_classes)
        self.fc     = BinaryTreeLayer(out_channel*8, num_classes,depth=depth)

        self.calibration_iter = calibration_iter

    def forward(self,x):
        x = self.bb(x)
        x = self.max(x)

        x = self.dfcsm1(x)
        x = self.max1(x)

        x = self.dfcsm2(x)
        x = self.max2(x)

        x = self.dfcsm3(x)
        x = self.max3(x)

        x = self.dfcsm4(x)
        x = self.max4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
    
        return x
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        # 初始化参数权重
        params = torch.ones_like(logits)

        # 迭代的真值发现算法
        for _ in range(self.calibration_iter):
            # 根据参数权重进行softmax
            probs = torch.nn.functional.softmax(logits * params, dim=1)

            # 计算平均概率
            avg_probs = torch.mean(probs, dim=0, keepdim=True)

            # 重新计算调整后的概率
            adjusted_probs = logits + torch.log(avg_probs)

            # 更新参数权重
            params = params * torch.exp(adjusted_probs - logits)

        # 根据最终的参数权重计算校准后的概率
        calibrated_probs = torch.nn.functional.softmax(logits * params, dim=1)

        return calibrated_probs  
        

def RFACANet(num_classes=1000, depth=3,calibration_iter=10,input_channel=3, out_channel=64, dm=[1,3,6,9], strides=1):
    return RFACA_Net(num_classes, depth, calibration_iter, input_channel, out_channel, dm, strides)

