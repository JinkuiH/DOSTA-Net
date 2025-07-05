"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""
from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from .decoder import UnetDecoder,UnetDecoderReturn
from .encoders import get_encoder
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)
        self.conv3 = conv(1, n_feat, kernel_size, bias=bias)
#x2_samfeats, stage1_img = self.sam12(res1[0], x3_img)
    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)


    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
 
        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)

        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]

##########################################################################
##---------- Resizing Modules ----------    
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                #nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False)
                                )

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class RDB(nn.Module):
    def __init__(self, in_channels, n_dense_layers, growth_rate, kernel_size=3, act=nn.PReLU(), bias=False):
        super(RDB, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(n_dense_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(current_channels, growth_rate, kernel_size, padding=kernel_size//2, bias=bias),
                act
            ))
            current_channels += growth_rate
        # Bottleneck layer to compress features back to original channels
        self.conv_1x1 = nn.Conv2d(current_channels, in_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            new_features = layer(out)
            out = torch.cat([out, new_features], dim=1)
        out = self.conv_1x1(out)
        return out + x  # Residual connection
    
##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        # self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        # self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        # self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb1 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        
    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x



# ##########################################################################
# #先融合高分辨率，再融合低分辨率
# class ORSNetV2(nn.Module):
#     def __init__(self, n_feat, encoder_ch, decoder_ch,scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
#         super(ORSNetV2, self).__init__()

#         self.orb1 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)
#         self.orb2 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)
#         self.orb3 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)

#         self.up_enc1 = UpSample(encoder_ch[0], n_feat)
#         self.up_dec1 = UpSample(decoder_ch[-2], n_feat)

#         self.up_enc2 = nn.Sequential(UpSample(encoder_ch[1], n_feat), UpSample(n_feat, n_feat))
#         self.up_dec2 = nn.Sequential(UpSample(decoder_ch[-3], n_feat), UpSample(n_feat, n_feat))

#         # self.conv_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
#         self.conv_enc2 = nn.Conv2d(encoder_ch[0], n_feat, kernel_size=1, bias=bias)
#         self.conv_enc3 = nn.Conv2d(encoder_ch[1], n_feat, kernel_size=1, bias=bias)

#         self.conv_dec1 = nn.Conv2d(decoder_ch[2], n_feat, kernel_size=1, bias=bias)
#         self.conv_dec2 = nn.Conv2d(decoder_ch[1], n_feat, kernel_size=1, bias=bias)
#         self.conv_dec3 = nn.Conv2d(decoder_ch[0], n_feat, kernel_size=1, bias=bias)
        
#     def forward(self, x, encoder_outs, decoder_outs):
#         x = self.orb1(x)
#         x = x +  self.conv_dec1(decoder_outs[0])

#         x = self.orb2(x)
#         x = x + self.conv_enc2(self.up_enc1(encoder_outs[0])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

#         x = self.orb3(x)
#         x = x + self.conv_enc3(self.up_enc2(encoder_outs[1])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

#         return x

class RDBNet(nn.Module):
    def __init__(self, n_feat, encoder_ch, decoder_ch, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab,
                 rdb_growth_rate=32, rdb_num_layers=8):  # 新增参数
        super(RDBNet, self).__init__()

        # 将ORB替换为RDB
        self.rdb1 = RDB(n_feat, n_dense_layers=rdb_num_layers, growth_rate=rdb_growth_rate, 
                       kernel_size=kernel_size, act=act, bias=bias)
        self.rdb2 = RDB(n_feat, n_dense_layers=rdb_num_layers, growth_rate=rdb_growth_rate, 
                       kernel_size=kernel_size, act=act, bias=bias)
        self.rdb3 = RDB(n_feat, n_dense_layers=rdb_num_layers, growth_rate=rdb_growth_rate, 
                       kernel_size=kernel_size, act=act, bias=bias)

        # 保持原有连接结构不变
        self.up_enc1 = UpSample(encoder_ch[0], n_feat)
        self.up_dec1 = UpSample(decoder_ch[-2], n_feat)

        self.up_enc2 = nn.Sequential(
            UpSample(encoder_ch[1], n_feat), 
            UpSample(n_feat, n_feat)
        )
        self.up_dec2 = nn.Sequential(
            UpSample(decoder_ch[-3], n_feat),
            UpSample(n_feat, n_feat)
        )

        self.conv_enc2 = nn.Conv2d(encoder_ch[0], n_feat, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(encoder_ch[1], n_feat, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(decoder_ch[2], n_feat, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(decoder_ch[1], n_feat, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(decoder_ch[0], n_feat, kernel_size=1, bias=bias)
        
    def forward(self, x, encoder_outs, decoder_outs):
        # 替换ORB为RDB的前向传播
        x = self.rdb1(x)
        x = x + self.conv_dec1(decoder_outs[0])

        x = self.rdb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[0])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.rdb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[1])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


class twoStageNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, n_feat=32, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(twoStageNet, self).__init__()

        act=nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        # self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.resConv = conv(n_feat, out_c, kernel_size, bias=bias)
        # self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.tail     = conv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, x3_img):
        # Original-resolution Image for Stage 3
       
        x1 = self.shallow_feat1(x3_img)
        
        ## Process features of all 4 patches with Encoder of Stage 1
        feat1 = self.stage1_encoder(x1)

        ## Pass features through Decoder of Stage 1
        res1 = self.stage1_decoder(feat1)
        res1_tail = self.resConv(res1[0])

        x2_input = x3_img + res1_tail
        ## Apply SAM
        # x2_samfeats, stage1_img = self.sam12(res1[0], x3_img)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2     = self.shallow_feat2(x2_input)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        # x2_cat = self.concat12(torch.cat([x2, x2_samfeats], 1))
        # print(feat1[2].shape, res1[2].shape)
        x2_cat = self.stage2_orsnet(x2, feat1, res1)

        stage2_img = self.tail(x2_cat)

        # return [stage2_img+x3_img, stage1_img]
        return stage2_img+x3_img



class twoStageNetSimpleUnet(nn.Module):
    def __init__(self, in_c=1, out_c=1, n_feat=32, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False,
                 encoder_name: str = "resnet34",
                encoder_depth: int = 5,
                encoder_weights: Optional[str] = "imagenet",
                decoder_use_batchnorm: bool = True,
                decoder_channels: List[int] = (256, 128, 64, 32, 16),
                decoder_attention_type: Optional[str] = None
        ):
        super(twoStageNetSimpleUnet, self).__init__()

        act=nn.PReLU()
        # self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))

    
        self.stage1_encoder = get_encoder(
            encoder_name,
            in_channels=in_c,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.stage1_decoder = UnetDecoderReturn(
            encoder_channels=self.stage1_encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
#    def __init__(self, n_feat, encoder_ch, decoder_ch,scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):

        # print('kkk:',self.stage1_encoder.out_channels, decoder_channels)
        self.stage2_orsnet = RDBNet(n_feat, self.stage1_encoder.out_channels[1:], decoder_channels,scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, 
                                      num_cab,rdb_growth_rate=32,  # 新增参数
                                      rdb_num_layers=4     # 对应原始num_cab
       )

        # self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.resConv = conv(decoder_channels[-1], out_c, kernel_size, bias=bias)
        # self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.tail     = conv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, x3_img):

        feat1 = self.stage1_encoder(x3_img)

        res1 = self.stage1_decoder(feat1)
        stage1_img = self.resConv(res1[0])

        x2_input = x3_img + stage1_img
        ## Apply SAM
        # x2_samfeats, stage1_img = self.sam12(res1[0], x3_img)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2     = self.shallow_feat2(x2_input)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        # x2_cat = self.concat12(torch.cat([x2, x2_samfeats], 1))
        # print(feat1[2].shape, res1[2].shape)
        x2_cat = self.stage2_orsnet(x2, feat1[1:], res1)

        stage2_img = self.tail(x2_cat)

        return [stage2_img+x3_img, stage1_img]
        # return stage2_img+x3_img

class twoStageUnetAttention(nn.Module):
    def __init__(self, in_c=1, out_c=1, n_feat=32, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False,
                 encoder_name: str = "resnet34",
                encoder_depth: int = 5,
                encoder_weights: Optional[str] = "imagenet",
                decoder_use_batchnorm: bool = True,
                decoder_channels: List[int] = (256, 128, 64, 32, 16),
                decoder_attention_type: Optional[str] = None
        ):
        super(twoStageUnetAttention, self).__init__()

        act=nn.PReLU()
        # self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))

    
        self.stage1_encoder = get_encoder(
            encoder_name,
            in_channels=in_c,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.stage1_decoder = UnetDecoderReturn(
            encoder_channels=self.stage1_encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
#    def __init__(self, n_feat, encoder_ch, decoder_ch,scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):

        # print('kkk:',self.stage1_encoder.out_channels, decoder_channels)
        self.stage2_rdbnet = RDBNet(n_feat, self.stage1_encoder.out_channels[1:], decoder_channels,scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, 
                                      num_cab,rdb_growth_rate=32,  # 新增参数
                                      rdb_num_layers=4     # 对应原始num_cab
       )

        self.sam12 = SAM(decoder_channels[-1], kernel_size=1, bias=bias)
        self.resConv = conv(decoder_channels[-1], out_c, kernel_size, bias=bias)
        self.concat12  = conv(96, n_feat, kernel_size, bias=bias)
        self.tail     = conv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, x3_img):

        feat1 = self.stage1_encoder(x3_img)

        res1 = self.stage1_decoder(feat1)
        stage1_img = self.resConv(res1[0])

        # x2_input = x3_img + stage1_img
        ## Apply SAM
        # x2_samfeats, stage1_img = self.sam12(res1[0], x3_img)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2     = self.shallow_feat2(x3_img+stage1_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        # x2_cat = self.concat12(torch.cat([x2, x2_samfeats], 1))
        # print(feat1[2].shape, res1[2].shape)
        x2_cat = self.stage2_rdbnet(x2, feat1[1:], res1)

        stage2_img = self.tail(x2_cat)

        return [stage2_img+x3_img, stage1_img+x3_img]
        # return stage2_img+x3_img