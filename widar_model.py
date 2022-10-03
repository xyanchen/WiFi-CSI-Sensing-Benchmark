import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class Widar_MLP(nn.Module):
    def __init__(self, num_classes):
        super(Widar_MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(22*20*20,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = x.view(-1,22*20*20)
        x = self.fc(x)
        return x

class Widar_LeNet(nn.Module):
    def __init__(self, num_classes):
        super(Widar_LeNet,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (22,20,20)
            nn.Conv2d(22,32,6,stride=2),
            nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=1),
            nn.ReLU(True),
            nn.Conv2d(64,96,3,stride=1),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(96*4*4,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,96*4*4)
        out = self.fc(x)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
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
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

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


        
        
class Widar_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes):
        super(Widar_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.ConvTranspose2d(22,3,7,stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3,3,kernel_size=7,stride=1),
            nn.ReLU()
        )
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
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
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    
def Widar_ResNet18(num_classes):
    return Widar_ResNet(Block, [2,2,2,2], num_classes = num_classes)
def Widar_ResNet50(num_classes):
    return Widar_ResNet(Bottleneck, [3,4,6,3], num_classes = num_classes)
    
def Widar_ResNet101(num_classes):
    return Widar_ResNet(Bottleneck, [3,4,23,3], num_classes = num_classes)


class Widar_RNN(nn.Module):
    def __init__(self,num_classes):
        super(Widar_RNN,self).__init__()
        self.rnn = nn.RNN(400,64,num_layers=1)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,22,400)
        x = x.permute(1,0,2)
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs

class Widar_GRU(nn.Module):
    def __init__(self,num_classes):
        super(Widar_GRU,self).__init__()
        self.gru = nn.GRU(400,64,num_layers=1)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,22,400)
        x = x.permute(1,0,2)
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return outputs

class Widar_LSTM(nn.Module):
    def __init__(self,num_classes):
        super(Widar_LSTM,self).__init__()
        self.lstm = nn.LSTM(400,64,num_layers=1)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,22,400)
        x = x.permute(1,0,2)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class Widar_BiLSTM(nn.Module):
    def __init__(self,num_classes):
        super(Widar_BiLSTM,self).__init__()
        self.lstm = nn.LSTM(400,64,num_layers=1,bidirectional=True)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,22,400)
        x = x.permute(1,0,2)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class Widar_CNN_GRU(nn.Module):
    def __init__(self,num_classes):
        super(Widar_CNN_GRU,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,8,6,2),
            nn.ReLU(),
            nn.Conv2d(8,16,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*3*3,64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64,64),
            nn.ReLU(),
        )
        self.gru = nn.GRU(64,128,num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128,num_classes),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        batch_size = len(x)
        # batch x 22 x 20 x 20
        x = x.view(batch_size*22,1,20,20)
        # 22*batch x 1 x 20 x 20
        x = self.encoder(x)
        # 22*batch x 16 x 3 x 3
        x = x.view(-1,16*3*3)
        x = self.fc(x)
        # 22*batch x 64
        x = x.view(-1,22,64)
        x = x.permute(1,0,2)
        # 22 x batch x 64
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return outputs

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 1, patch_size_w = 2, patch_size_h = 40, emb_size = 2*40, img_size = 22*400):
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size = (patch_size_w, patch_size_h), stride = (patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.position = nn.Parameter(torch.randn(int(img_size/emb_size) + 1, emb_size))
    
    def forward(self, x):
        x = x.view(-1,1,22,400)
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 80, num_heads = 5, dropout = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4, drop_p = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size = 80,
                 drop_p = 0.5,
                 forward_expansion = 4,
                 forward_drop_p = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth = 1, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, num_classes))
        
class Widar_ViT(nn.Sequential):
    def __init__(self,     
                in_channels = 1,
                patch_size_w = 2,
                patch_size_h = 40,
                emb_size = 80,
                img_size = 22*400,
                depth = 1,
                *,
                num_classes,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )
