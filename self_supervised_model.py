import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F


class MLP_Parrallel(nn.Module):
    def __init__(self):
        super(MLP_Parrallel, self).__init__()
        self.encoder_1 = MLP_encoder()
        self.encoder_2 = MLP_encoder()
        
        self.classifier = nn.Linear(128,14)

    def forward(self, x1, x2, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder_1(x1, flag=flag)
            x2 = self.encoder_2(x2, flag=flag)
            y1 = self.classifier(x1)
            y2 = self.classifier(x2)
            return y1, y2
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2

class MLP_encoder(nn.Module):
    def __init__(self,hidden_states = 256):
        super(MLP_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*114*500,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
        )
        self.mapping = nn.Linear(128, hidden_states)
            
        self.bn = nn.BatchNorm1d(hidden_states)

    def forward(self, x, flag='unsupervised'):
        x = x.view(-1, 3*114*500)
        x = self.encoder(x)
        if flag == 'supervised':
            return x
        else:
            x = self.bn(self.mapping(x))
            return x



class CNN_Parrallel(nn.Module):
    def __init__(self):
        super(CNN_Parrallel, self).__init__()
        self.encoder_1 = CNN_encoder()
        self.encoder_2 = CNN_encoder() 

        self.classifier = nn.Sequential(
            nn.Linear(96*4*6,128),
            nn.ReLU(),
            nn.Linear(128,14)
        )

    def forward(self, x1, x2, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder_1(x1, flag=flag)
            x2 = self.encoder_2(x2, flag=flag)
            y1 = self.classifier(x1)
            y2 = self.classifier(x2)
            return y1, y2
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2

class CNN_encoder(nn.Module):
    def __init__(self,hidden_states = 256):
        super(CNN_encoder, self).__init__()
        self.encoder = nn.Sequential(
            #input size: (3,114,500)
            nn.Conv2d(3,32,(15,23),stride=9),
            nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=(1,3)),
            nn.ReLU(True),
            nn.Conv2d(64,96,(7,3),stride=(1,3)),
            nn.ReLU(True),
        )
        self.mapping = nn.Linear(96*4*6, hidden_states)
            
        self.bn = nn.BatchNorm1d(hidden_states)
        
        self.conv_channel = 96
        self.conv_feat_num = 24

    def forward(self, x, flag='unsupervised'):
        x = self.encoder(x)
        # classifier
        x = x.view(-1, self.conv_channel*self.conv_feat_num)
        
        if flag == 'supervised':
            return x
        else:
            x = self.bn(self.mapping(x))
            return x



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

class ResNet_Parrallel(nn.Module):
    def __init__(self,ResBlock, layer_list):
        super(ResNet_Parrallel, self).__init__()
        self.encoder_1 = ResNet_encoder(ResBlock, layer_list)
        self.encoder_2 = ResNet_encoder(ResBlock, layer_list) # output: after BN
        
        self.classifier = nn.Linear(512*ResBlock.expansion, 14)

    def forward(self, x1, x2, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder_1(x1, flag=flag)
            x2 = self.encoder_2(x2, flag=flag)
            y1 = self.classifier(x1)
            y2 = self.classifier(x2)
            return y1, y2
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2

class ResNet_encoder(nn.Module):
    def __init__(self, ResBlock, layer_list, hidden_states = 256):
        super(ResNet_encoder, self).__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(3,3,(15,23),stride=(3,9)),
            nn.ReLU(),
            nn.Conv2d(3,3,kernel_size=(3,23),stride=1),
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
        
        
        self.mapping = nn.Linear(512*ResBlock.expansion, hidden_states)
            
        self.bn = nn.BatchNorm1d(hidden_states)
        

    def forward(self, x, flag='unsupervised'):
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # classifier
        x = x.reshape(x.shape[0], -1)
        
        if flag == 'supervised':
            return x
        else:
            x = self.bn(self.mapping(x))
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

def ResNet18_Parrallel():
    return ResNet_Parrallel(Block, [2,2,2,2])
def ResNet50_Parrallel():
    return ResNet_Parrallel(Bottleneck, [3,4,6,3])
def ResNet101_Parrallel():
    return ResNet_Parrallel(Bottleneck, [3,4,23,3])



class RNN_Parrallel(nn.Module):
    def __init__(self):
        super(RNN_Parrallel, self).__init__()
        self.encoder_1 = RNN_encoder()
        self.encoder_2 = RNN_encoder()
        
        self.classifier = nn.Linear(64,14)

    def forward(self, x1, x2, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder_1(x1, flag=flag)
            x2 = self.encoder_2(x2, flag=flag)
            y1 = self.classifier(x1)
            y2 = self.classifier(x2)
            return y1, y2
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2

class RNN_encoder(nn.Module):
    def __init__(self,hidden_states = 256):
        super(RNN_encoder, self).__init__()
        self.encoder = nn.RNN(342,64,num_layers=1)
        self.mapping = nn.Linear(64, hidden_states)
            
        self.bn = nn.BatchNorm1d(hidden_states)
        

    def forward(self, x, flag='unsupervised'):
        x = x.view(-1,342,500)
        x = x.permute(2,0,1)
        _, ht = self.encoder(x)
        # classifier
        x = ht[-1]
        
        if flag == 'supervised':
            return x
        else:
            x = self.bn(self.mapping(x))
            return x



class GRU_Parrallel(nn.Module):
    def __init__(self):
        super(GRU_Parrallel, self).__init__()
        self.encoder_1 = GRU_encoder()
        self.encoder_2 = GRU_encoder()
        
        self.classifier = nn.Linear(64,14)

    def forward(self, x1, x2, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder_1(x1, flag=flag)
            x2 = self.encoder_2(x2, flag=flag)
            y1 = self.classifier(x1)
            y2 = self.classifier(x2)
            return y1, y2
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2

class GRU_encoder(nn.Module):
    def __init__(self,hidden_states = 256):
        super(GRU_encoder, self).__init__()
        self.encoder = nn.GRU(342,64,num_layers=1)
        self.mapping = nn.Linear(64, hidden_states)
            
        self.bn = nn.BatchNorm1d(hidden_states)
        

    def forward(self, x, flag='unsupervised'):
        x = x.view(-1,342,500)
        x = x.permute(2,0,1)
        _, ht = self.encoder(x)
        # classifier
        x = ht[-1]
        
        if flag == 'supervised':
            return x
        else:
            x = self.bn(self.mapping(x))
            return x



class LSTM_Parrallel(nn.Module):
    def __init__(self):
        super(LSTM_Parrallel, self).__init__()
        self.encoder_1 = LSTM_encoder()
        self.encoder_2 = LSTM_encoder()
        
        self.classifier = nn.Linear(64,14)

    def forward(self, x1, x2, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder_1(x1, flag=flag)
            x2 = self.encoder_2(x2, flag=flag)
            y1 = self.classifier(x1)
            y2 = self.classifier(x2)
            return y1, y2
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2

class LSTM_encoder(nn.Module):
    def __init__(self,hidden_states = 256):
        super(LSTM_encoder, self).__init__()
        self.encoder = nn.LSTM(342,64,num_layers=1)
        self.mapping = nn.Linear(64, hidden_states)
            
        self.bn = nn.BatchNorm1d(hidden_states)
        

    def forward(self, x, flag='unsupervised'):
        x = x.view(-1,342,500)
        x = x.permute(2,0,1)
        _, (ht,ct) = self.encoder(x)
        # classifier
        x = ht[-1]
        
        if flag == 'supervised':
            return x
        else:
            x = self.bn(self.mapping(x))
            return x



class BiLSTM_Parrallel(nn.Module):
    def __init__(self):
        super(BiLSTM_Parrallel, self).__init__()
        self.encoder_1 = BiLSTM_encoder()
        self.encoder_2 = BiLSTM_encoder()
        
        self.classifier = nn.Linear(64,14)

    def forward(self, x1, x2, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder_1(x1, flag=flag)
            x2 = self.encoder_2(x2, flag=flag)
            y1 = self.classifier(x1)
            y2 = self.classifier(x2)
            return y1, y2
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2

class BiLSTM_encoder(nn.Module):
    def __init__(self,hidden_states = 256):
        super(BiLSTM_encoder, self).__init__()
        self.encoder = nn.LSTM(342,64,num_layers=1,bidirectional=True)
        self.mapping = nn.Linear(64, hidden_states)
            
        self.bn = nn.BatchNorm1d(hidden_states)
        

    def forward(self, x, flag='unsupervised'):
        x = x.view(-1,342,500)
        x = x.permute(2,0,1)
        _, (ht,ct) = self.encoder(x)
        # classifier
        x = ht[-1]
        
        if flag == 'supervised':
            return x
        else:
            x = self.bn(self.mapping(x))
            return x



class CNN_GRU_Parrallel(nn.Module):
    def __init__(self):
        super(CNN_GRU_Parrallel, self).__init__()
        self.encoder_1 = CNN_GRU_encoder()
        self.encoder_2 = CNN_GRU_encoder()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128,14),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder_1(x1, flag=flag)
            x2 = self.encoder_2(x2, flag=flag)
            y1 = self.classifier(x1)
            y2 = self.classifier(x2)
            return y1, y2
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2

class CNN_GRU_encoder(nn.Module):
    def __init__(self,hidden_states = 256):
        super(CNN_GRU_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,16,12,6),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,7,3),
            nn.ReLU(),
        )
        self.mean = nn.AvgPool1d(32)
        self.gru = nn.GRU(8,128,num_layers=1)
        
        self.mapping = nn.Linear(128, hidden_states)
            
        self.bn = nn.BatchNorm1d(hidden_states)
        

    def forward(self, x, flag='unsupervised'):
        batch_size = len(x)
        # batch x 3 x 114 x 500
        x = x.view(batch_size,3*114,500)
        x = x.permute(0,2,1)
        # batch x 500 x 342
        x = x.reshape(batch_size*500,1, 3*114)
        # (batch x 500) x 1 x 342
        x = self.encoder(x)
        # (batch x 500) x 32 x 8
        # try 32, (32x8)
        x = x.permute(0,2,1)
        x = self.mean(x)
        x = x.reshape(batch_size, 500, 8)
        # batch x 500 x 8
        x = x.permute(1,0,2)
        # 500 x batch x 8
        _, ht = self.gru(x)
        # classifier
        x = ht[-1]
        
        if flag == 'supervised':
            return x
        else:
            x = self.bn(self.mapping(x))
            return x



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 1, patch_size_w = 9, patch_size_h = 25, emb_size = 9*25, img_size = 342*500):
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
        x = x.view(-1,1,342,500)
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 225, num_heads = 5, dropout = 0.0):
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
                 emb_size = 225,
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
        
class ViTEncoder(nn.Sequential):
    def __init__(self,     
                in_channels = 1,
                patch_size_w = 9,
                patch_size_h = 25,
                emb_size = 225,
                img_size = 342*500,
                depth = 1,
                *,
                num_classes = 14,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        )

class ViT_Parrallel(nn.Module):
    def __init__(self,
                emb_size = 225,
                num_classes = 14,
                ):
        super(ViT_Parrallel, self).__init__()
        self.encoder_1 = ViT_encoder()
        self.encoder_2 = ViT_encoder()
        
        self.classifier = ClassificationHead(emb_size, num_classes)

    def forward(self, x1, x2, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder_1(x1, flag=flag)
            x2 = self.encoder_2(x2, flag=flag)
            y1 = self.classifier(x1)
            y2 = self.classifier(x2)
            return y1, y2
        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        return x1, x2

class ViT_encoder(nn.Module):
    def __init__(self,hidden_states = 256):
        super(ViT_encoder, self).__init__()
        self.encoder = ViTEncoder()
        
        self.mapping = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.Linear(225, hidden_states)
        )
            
        self.bn = nn.BatchNorm1d(hidden_states)
        

    def forward(self, x, flag='unsupervised'):
        x = self.encoder(x)
        if flag == 'supervised':
            return x
        else:
            x = self.bn(self.mapping(x))
            return x
