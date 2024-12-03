import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, groups=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, groups=groups)
        self.batch_norm3 = nn.BatchNorm1d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, groups=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, groups=groups)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


        
        
class NTU_Fi_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, groups=32, width_per_group=4):
        super(NTU_Fi_ResNet, self).__init__()
        # self.reshape = nn.Sequential(
        #     nn.Conv1d(3,3,23,stride=9),
        #     nn.ReLU(),
        #     nn.Conv1d(3,3,kernel_size=23,stride=1),
        #     nn.ReLU()
        # )
        self.reshape = nn.Sequential(
            nn.Conv1d(3,3,15,stride=3),
            nn.ReLU(),
            nn.Conv1d(3,3,kernel_size=3,stride=1),
            nn.ReLU()
        )
        self.in_channels = 64
        self.groups = groups
        self.width_per_group = width_per_group
        
        self.conv1 = nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride, groups=self.groups),
                nn.BatchNorm1d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride, groups=self.groups))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes, groups=self.groups))
            
        return nn.Sequential(*layers)
    
def oneD_lw_ResNet18(num_classes):
    return NTU_Fi_ResNet(Block, [2,2,2,2], num_classes = num_classes, groups=32, width_per_group=4)

def oneD_lw_ResNet50(num_classes):
    return NTU_Fi_ResNet(Bottleneck, [3,4,6,3], num_classes = num_classes, groups=32, width_per_group=4)

def oneD_lw_ResNet101(num_classes):
    return NTU_Fi_ResNet(Bottleneck, [3,4,23,3], num_classes = num_classes, groups=32, width_per_group=4)
