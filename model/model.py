import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        se_weight = self.global_avg_pool(x).view(batch, channels)
        se_weight = self.fc(se_weight).view(batch, channels, 1, 1)
        return x * se_weight


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        self.se_block = SEBlock(dim_out)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        x = self.se_block(x)
        return x + residual

class AdaptiveNoiseLayer(nn.Module):
    def __init__(self, num_channels, training=True, mean=0.0, std=0.1, apply_noise=True):
        super(AdaptiveNoiseLayer, self).__init__()
        self.training = training
        self.apply_noise = apply_noise
        self.weight = nn.Parameter(torch.full((num_channels,), std))  # 初始化為 std
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training and self.apply_noise:
            noise = torch.normal(mean=self.mean, std=self.std, size=x.shape, device=x.device)
            return x + self.weight.view(1, -1, 1, 1) * noise
        return x


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = max(256, norm_nc // 2)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(nhidden),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        segmap = segmap.float()
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear', align_corners=False)

        normalized = self.norm(x)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return normalized * (1 + gamma) + beta



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, label_nc, training):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.spade  = SPADE(out_channels, label_nc)
        self.noise  = AdaptiveNoiseLayer(out_channels, training)

    def forward(self, x, segmap):
        x = self.deconv(x)
        x = F.leaky_relu(self.spade(x, segmap))
        x = self.noise(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, training):
        super().__init__()
        self.conv  = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.noise = AdaptiveNoiseLayer(out_channels, training)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x))
        x = self.noise(x)
        return x

class Encoder_Decoder(nn.Module):
    def __init__(self, curr_dim,label_nc, training, num_layers = 3, base_channels= 64, repeat_num = 6  ):
        super().__init__()

        self.encoder_layers = nn.ModuleList()   
        curr_dim = curr_dim
        channels = []  
        for i in range(num_layers):

            self.encoder_layers.append(EncoderBlock(curr_dim, base_channels, training))
            channels.append(base_channels)
            curr_dim = base_channels
            base_channels *= 2 

        for i in range(repeat_num):
            self.encoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))           
        self.decoder_layers = nn.ModuleList()

        channels.reverse()
        #decoder

        for  i in range(num_layers - 1):

            self.decoder_layers.append(DecoderBlock(curr_dim, channels[i+1], label_nc, training=training))
            curr_dim = channels[i+1]
        
        self.output_layer = nn.ConvTranspose2d(curr_dim, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.md_output_layer = nn.ConvTranspose2d(curr_dim, 1, kernel_size=4, stride=2, padding=1)
        self.spatial_activation = nn.Tanh()


    def forward(self, x, segmap):

        for block in self.encoder_layers:
            x = block(x)

        for block in self.decoder_layers:

            x = block(x, segmap)

        foreground  = self.output_layer(x)
        foreground  = self.tanh(foreground)

        spatial_map = self.md_output_layer(x)
        spatial_map = self.spatial_activation(spatial_map)
        spatial_map = (spatial_map + 1 )/ 2
        return foreground  , spatial_map
    
class Foreground(nn.Module):
    def __init__(self,curr_dim, label_nc,num_layers, base_channels, repeat_num, training):
        super(Foreground, self).__init__()
        self.encoder_decoder = Encoder_Decoder(curr_dim=curr_dim,
                                            label_nc=label_nc, 
                                            num_layers=num_layers, 
                                            base_channels=base_channels,
                                            repeat_num=repeat_num, 
                                            training=training)  
    
    def forward(self, sample, segmap):

        foreground , spatial_map= self.encoder_decoder(sample, segmap)

        return foreground, spatial_map

def Layer_wise_composition(sample,Spatial_map,foreground):
        Spatial_map_expanded = Spatial_map.expand_as(foreground)

        return sample * (1 - Spatial_map_expanded) + foreground * Spatial_map_expanded
        
class Generator(nn.Module):
    def __init__(self,curr_dim, label_nc, num_layers, base_channels, repeat_num, training = True):
        super().__init__()
        self.foreground = Foreground(curr_dim, label_nc, num_layers, base_channels, repeat_num, training)

    def forward(self, normal_sample, control_map):
        
        defect_foreground, defect_spatial_map = self.foreground(normal_sample, control_map)

        synthesized_defect_sample = Layer_wise_composition(sample=normal_sample,Spatial_map=defect_spatial_map,foreground=defect_foreground)

        reverse_control_map = 1 - control_map  

        restore_foreground, restore_spatial_map = self.foreground(synthesized_defect_sample, reverse_control_map)

        synthesized_restore_sample = Layer_wise_composition(sample=synthesized_defect_sample,Spatial_map=restore_spatial_map,foreground=restore_foreground)

        return synthesized_restore_sample, synthesized_defect_sample, defect_spatial_map, restore_spatial_map


class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=3):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv_src = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)  # 判別真假
        self.conv_cls = nn.Conv2d(curr_dim, c_dim, kernel_size=3, stride=1, padding=1, bias=False)  # 輸出分類

        # 添加上採樣層，將空間尺寸恢復到原始大小
        self.upsample = nn.ConvTranspose2d(c_dim, c_dim, kernel_size=16, stride=8, padding=4, output_padding=0, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_src(h)  # 輸出真假：[B, 1, H, W]
        out_cls = self.conv_cls(h)  # 輸出分類：[B, C, H, W]
        out_cls = self.upsample(out_cls)  # 上採樣：[B, C, 128, 128]
        return out_src, out_cls
