import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class UT_HAR_MLP(nn.Module):
    def __init__(self):
        super(UT_HAR_MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(250*90,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,7)
        )
        
    def forward(self,x):
        x = x.view(-1,250*90)
        x = self.fc(x)
        return x


class UT_HAR_LeNet(nn.Module):
    def __init__(self):
        super(UT_HAR_LeNet,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (1,250,90)
            nn.Conv2d(1,32,7,stride=(3,1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,(5,4),stride=(2,2),padding=(1,0)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,96,(3,3),stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(96*4*4,128),
            nn.ReLU(),
            nn.Linear(128,7)
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
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
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
    

class UT_HAR_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=7):
        super(UT_HAR_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(1,3,7,stride=(3,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3,3,kernel_size=(10,11),stride=1),
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

def UT_HAR_ResNet18():
    return UT_HAR_ResNet(Block, [2,2,2,2])
def UT_HAR_ResNet50():
    return UT_HAR_ResNet(Bottleneck, [3,4,6,3])
def UT_HAR_ResNet101():
    return UT_HAR_ResNet(Bottleneck, [3,4,23,3])


class UT_HAR_RNN(nn.Module):
    def __init__(self,hidden_dim=64):
        super(UT_HAR_RNN,self).__init__()
        self.rnn = nn.RNN(90,hidden_dim,num_layers=1)
        self.fc = nn.Linear(hidden_dim,7)
    def forward(self,x):
        x = x.view(-1,250,90)
        x = x.permute(1,0,2)
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_GRU(nn.Module):
    def __init__(self,hidden_dim=64):
        super(UT_HAR_GRU,self).__init__()
        self.gru = nn.GRU(90,hidden_dim,num_layers=1)
        self.fc = nn.Linear(hidden_dim,7)
    def forward(self,x):
        x = x.view(-1,250,90)
        x = x.permute(1,0,2)
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_LSTM(nn.Module):
    def __init__(self,hidden_dim=64):
        super(UT_HAR_LSTM,self).__init__()
        self.lstm = nn.LSTM(90,hidden_dim,num_layers=1)
        self.fc = nn.Linear(hidden_dim,7)
    def forward(self,x):
        x = x.view(-1,250,90)
        x = x.permute(1,0,2)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_BiLSTM(nn.Module):
    def __init__(self,hidden_dim=64):
        super(UT_HAR_BiLSTM,self).__init__()
        self.lstm = nn.LSTM(90,hidden_dim,num_layers=1,bidirectional=True)
        self.fc = nn.Linear(hidden_dim,7)
    def forward(self,x):
        x = x.view(-1,250,90)
        x = x.permute(1,0,2)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_CNN_GRU(nn.Module):
    def __init__(self):
        super(UT_HAR_CNN_GRU,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (250,90)
            nn.Conv1d(250,250,12,3),
            nn.ReLU(True),
            nn.Conv1d(250,250,5,2),
            nn.ReLU(True),
            nn.Conv1d(250,250,5,1)
            # 250 x 8
        )
        self.gru = nn.GRU(8,128,num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128,7),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        # batch x 1 x 250 x 90
        x = x.view(-1,250,90)
        x = self.encoder(x)
        # batch x 250 x 8
        x = x.permute(1,0,2)
        # 250 x batch x 8
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return outputs


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 1, patch_size_w = 50, patch_size_h = 18, emb_size = 50*18, img_size = 250*90):
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
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 900, num_heads = 5, dropout = 0.0):
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
                 emb_size = 900,
                 drop_p = 0.,
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
    def __init__(self, emb_size = 900, n_classes = 7):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
        
class UT_HAR_ViT(nn.Sequential):
    def __init__(self,     
                in_channels = 1,
                patch_size_w = 50,
                patch_size_h = 18,
                emb_size = 900,
                img_size = 250*90,
                depth = 1,
                n_classes = 7,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
