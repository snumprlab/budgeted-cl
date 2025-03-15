import torch.nn as nn
from models.layers import ConvBlock, InitialBlock, FinalBlock
import copy
import torch

class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super(SparsifyBase, self).__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()

        return hook

    def record_activation(self):
        self.register_forward_hook(self.get_activation())

class Sparsify2D_all(SparsifyBase):
    """cross channel sparsify"""

    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_all, self).__init__()
        self.sr = sparse_ratio
        self.act_count = None

    def update_count(self, mask):
        if self.act_count is None:
            self.act_count = mask
        else:
            self.act_count += mask

    def forward(self, x):
        size = x.shape[1]
        k = int(self.sr * size)
        tmpx = x.view(x.shape[0], x.shape[1], -1)
        topvalues = tmpx.topk(1, dim=2)[0].squeeze(2)
        topval, indices = topvalues.topk(k, dim=1)
        act_count = torch.zeros(topvalues.shape, device=topvalues.device)
        self.update_count(act_count.scatter_(1, indices, 1).sum(dim=0))
        comp = act_count.unsqueeze(2).repeat(1, 1, tmpx.shape[2]).view_as(x)
        return comp * x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, opt, inChannels, outChannels, stride=1, downsample=None, kwinner=False):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        expansion = 1
        self.conv1 = ConvBlock(
            opt=opt,
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv2 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels * expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            last=True
        )
        self.activate = getattr(nn, opt.activetype)()
        self.kwinner = kwinner
        if self.kwinner:
            self.kwinner1 = Sparsify2D_all(0.9)
            self.kwinner2 = Sparsify2D_all(0.9)  # Sparsify2D()

        self.shortcut = nn.Sequential()
        if self.downsample is not None:
            self.shortcut = nn.Sequential(self.downsample)

    def forward(self, input_list):
        x, features, get_features, detached = input_list
        _out = self.conv1(x)
        if self.kwinner:
            _out = self.kwinner1(_out)
        _out = self.conv2(_out)
        shortcut = self.shortcut(x)
        
        _out = self.activate(_out) + shortcut
        if self.kwinner:
            _out = self.kwinner2(_out)

        if get_features:
            if detached:
                # d_out = self.conv1(x.detach())
                d_out = self.conv1(x)
                d_out = self.conv2(d_out)
                if self.downsample is not None:
                    # d_shortcut = self.downsample(x.detach())
                    d_shortcut = self.downsample(x)
                else:
                    d_shortcut = x.detach()
                d_out = d_out + d_shortcut
                features.append(d_out)
            else:
                features.append(_out)
        return [_out, features, get_features, detached]


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, opt, inChannels, outChannels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        expansion = 4
        self.conv1 = ConvBlock(
            opt=opt,
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv3 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels * expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.downsample = downsample

    def forward(self, input_list):
        x, features, get_features, detached = input_list
        _out = self.conv1(x)
        _out = self.conv2(_out)
        _out = self.conv3(_out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        _out = _out + shortcut
        if get_features:
            if detached:
                d_out = self.conv1(x.detach())
                d_out = self.conv2(d_out)
                d_out = self.conv3(d_out)
                if self.downsample is not None:
                    d_shortcut = self.downsample(x.detach())
                else:
                    d_shortcut = x.detach()
                d_out = d_out + d_shortcut
                features.append(d_out)
            else:
                features.append(_out)
        return [_out, features, get_features, detached]


class ResidualBlock(nn.Module):
    def __init__(self, opt, block, inChannels, outChannels, depth, stride=1, kwinner = False):
        super(ResidualBlock, self).__init__()
        if stride != 1 or inChannels != outChannels * block.expansion:
            downsample = ConvBlock(
                opt=opt,
                in_channels=inChannels,
                out_channels=outChannels * block.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            downsample = None
        self.blocks = nn.Sequential()
        self.blocks.add_module(
            "block0", block(opt, inChannels, outChannels, stride, downsample, kwinner = kwinner)
        )
        inChannels = outChannels * block.expansion
        for i in range(1, depth):
            self.blocks.add_module(
                "block{}".format(i), block(opt, inChannels, outChannels, kwinner = kwinner)
            )

    def forward(self, x, features=None, get_features=False, detached=False):
        return self.blocks([x, features, get_features, detached])[:2]


class ResNet(nn.Module):
    def __init__(self, opt, model_imagenet=False, channel_constant=1, kwinner=False):
        super(ResNet, self).__init__()
        depth = opt.depth
        # self.quant = torch.ao.quantization.QuantStub()
        # self.dequant = torch.ao.quantization.DeQuantStub()
        self.model_imagenet = model_imagenet
        if depth in [20, 32, 44, 56, 110, 1202]:
            blocktype, self.nettype = "BasicBlock", "cifar"
        elif depth in [164, 1001]:
            blocktype, self.nettype = "BottleneckBlock", "cifar"
        elif depth in [18, 34]:
            blocktype, self.nettype = "BasicBlock", "imagenet"
        elif depth in [50, 101, 152]:
            blocktype, self.nettype = "BottleneckBlock", "imagenet"
        assert depth in [20, 32, 44, 56, 110, 1202, 164, 1001, 18, 34, 50, 101, 152]

        if blocktype == "BasicBlock" and self.nettype == "cifar":
            assert (
                depth - 2
            ) % 6 == 0, (
                "Depth should be 6n+2, and preferably one of 20, 32, 44, 56, 110, 1202"
            )
            n = (depth - 2) // 6
            block = BasicBlock
            in_planes, out_planes = 16, 64
        elif blocktype == "BottleneckBlock" and self.nettype == "cifar":
            assert (
                depth - 2
            ) % 9 == 0, "Depth should be 9n+2, and preferably one of 164 or 1001"
            n = (depth - 2) // 9
            block = BottleneckBlock
            in_planes, out_planes = 16, 64
        elif blocktype == "BasicBlock" and self.nettype == "imagenet":
            assert depth in [18, 34]
            num_blocks = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
            block = BasicBlock
            in_planes, out_planes = 64, 512  # 20, 160
        elif blocktype == "BottleneckBlock" and self.nettype == "imagenet":
            assert depth in [50, 101, 152]
            if depth == 50:
                num_blocks = [3, 4, 6, 3]
            elif depth == 101:
                num_blocks = [3, 4, 23, 3]
            elif depth == 152:
                num_blocks = [3, 8, 36, 3]
            block = BottleneckBlock
            in_planes, out_planes = 64, 512
        else:
            assert 1 == 2
        
        self.num_classes = opt.num_classes
        if model_imagenet:
            self.initial = InitialBlock(
                opt=opt, out_channels=in_planes, kernel_size=7, stride=2, padding=3
            )            
            self.maxpool = nn.MaxPool2d(kernel_size=3,  stride=2, padding=1)
        else:
            self.initial = InitialBlock(
                opt=opt, out_channels=in_planes, kernel_size=3, stride=1, padding=1
            )
        if self.nettype == "cifar":
            self.group1 = ResidualBlock(opt, block, 16, int(16 * channel_constant), n, stride=1, kwinner=kwinner)
            self.group2 = ResidualBlock(opt, block, int(16 * block.expansion * channel_constant), int(32 * channel_constant), n, stride=2, kwinner=kwinner)
            self.group3 = ResidualBlock(opt, block, int(32 * block.expansion * channel_constant), int(64 * channel_constant), n, stride=2, kwinner=kwinner)
        elif self.nettype == "imagenet":
            self.group1 = ResidualBlock(
                opt, block, 64, 64, num_blocks[0], stride=1, kwinner=kwinner
            )  # For ResNet-S, convert this to 20,20
            self.group2 = ResidualBlock(
                opt, block, 64 * block.expansion, 128, num_blocks[1], stride=2, kwinner=kwinner
            )  # For ResNet-S, convert this to 20,40
            self.group3 = ResidualBlock(
                opt, block, 128 * block.expansion, 256, num_blocks[2], stride=2, kwinner=kwinner
            )  # For ResNet-S, convert this to 40,80
            self.group4 = ResidualBlock(
                opt, block, 256 * block.expansion, 512, num_blocks[3], stride=2, kwinner=kwinner
            )  # For ResNet-S, convert this to 80,160
        else:
            assert 1 == 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dim_out = int(out_planes * block.expansion * channel_constant)
        #self.fc = FinalBlock(opt=opt, in_channels=out_planes * block.expansion)
        self.fc = nn.Linear(int(out_planes * block.expansion * channel_constant), self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def forward(self, x, get_feature=False, get_features=False, detached=False, get_features_detach=True, include_out_for_features=False, get_block_features=False, quantization=False):
        last_layer_features = []
        features = []
        block_features = []
        # if quantization:
        #     x = self.quant(x)
        out_init = self.initial(x)
        if self.model_imagenet:
            out_init = self.maxpool(out_init)
        
        if get_features:
            features.append(out_init)
            #print("len0", len(last_layer_features))
            if get_features_detach: last_layer_features.append(features[-1].detach())
            else: last_layer_features.append(features[-1])
            block_features.append(out_init)
        features = []
        out1, features = self.group1(out_init, features, get_features, detached)
        
        if get_features:
            #print("len1", len(last_layer_features))
            if get_features_detach: last_layer_features.append(features[-1].detach())
            else: last_layer_features.append(features[-1])
            block_features.append(features)
        features = []
        out2, features = self.group2(out1, features, get_features, detached)
        
        if get_features:
            #print("len2", len(last_layer_features))
            if get_features_detach: last_layer_features.append(features[-1].detach())
            else: last_layer_features.append(features[-1])
            block_features.append(features)
        features = []
        out3, features = self.group3(out2, features, get_features, detached)
        
        if get_features:
            #print("len3", len(last_layer_features))
            if get_features_detach: last_layer_features.append(features[-1].detach())
            else: last_layer_features.append(features[-1])
            block_features.append(features)

        if self.nettype == "imagenet":
            features = []
            out4, features = self.group4(out3, features, get_features, detached)
            if get_features:
                if get_features_detach: last_layer_features.append(features[-1].detach())
                else: last_layer_features.append(features[-1])
                
            feature = self.pool(out4)
        else:
            feature = self.pool(out3)
            
        feature = feature.view(x.size(0), -1)
        out = self.fc(feature)

        # if quantization:
        #     out = self.dequant(x)

        if include_out_for_features:
            last_layer_features.append(out)
        
        if get_feature:
            return out, feature
        elif get_features:
            if get_block_features:
                return out, block_features
            return out, last_layer_features
        else:
            return out

class ResNet_G(ResNet):
    def __init__(self, opt, model_imagenet=False, channel_constant=1, kwinner=False):
        super().__init__(opt, model_imagenet)
        self.model_imagenet = model_imagenet
        del self.fc, self.pool
        
        
        self.ver2 = opt["ver2"]
        
        if self.ver2 and self.nettype == "cifar":
            self.num_channels = self.group2.blocks[-1].conv2.out_channels
            del self.group3
        else:
            self.num_channels = self.group3.blocks[-1].conv2.out_channels
        
        if self.nettype == "cifar" and not self.ver2:
            self.blocks = nn.Sequential()
            for i in range(3):
                self.blocks.add_module(
                    f"block{i}", self.group3.blocks[i]
                )
            self.group3.blocks = copy.deepcopy(self.blocks)
        if self.nettype == "imagenet":
            del self.group4

    def forward(self, x, get_feature=False, get_features=False, detached=False):
        features = []
        x = self.initial(x)
        if self.model_imagenet:
            x = self.maxpool(x)
        x, features = self.group1(x, features, get_features, detached)
        out, features = self.group2(x, features, get_features, detached)
        if self.ver2 and self.nettype == "cifar":
            return out
        out, features = self.group3(out, features, get_features, detached)
        return out
    
class ResNet_F(ResNet):
    def __init__(self, opt, model_imagenet=False, channel_constant=1, kwinner=False):
        super().__init__(opt, model_imagenet)
        del self.initial, self.group1, self.group2
        self.ver2 = opt["ver2"]
        
        if not self.ver2 and self.nettype == "cifar":
            self.blocks = nn.Sequential()
            for i in range(3,5):
                self.blocks.add_module(
                    f"block{i}", self.group3.blocks[i]
                )
            self.group3.blocks = copy.deepcopy(self.blocks)
            
        if self.nettype == "imagenet":
            del self.group3

            
    def forward(self, x, get_feature=False, get_features=False, detached=False):
        features = []
        if self.nettype == "cifar":
            out1, features = self.group3(x, features, get_features, detached)
        elif self.nettype == "imagenet":
            out1, features = self.group4(x, features, get_features, detached)
        feature = self.pool(out1)
        feature = feature.view(x.size(0), -1)
        out = self.fc(feature)
        return out