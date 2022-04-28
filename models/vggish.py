import torch
import torch.nn as nn
from torch import hub
from tqdm import tqdm

VGGISH_WEIGHTS = "https://github.com/harritaylor/torchvggish/" \
    "releases/download/v0.1/vggish-10086976.pth"
PCA_PARAMS = "https://github.com/harritaylor/torchvggish/" \
    "releases/download/v0.1/vggish_pca_params-970ea276.pth"


class VGG(nn.Module):
    def __init__(self, features, cuda=True, num_classes=128):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feat=False):
        # import pdb; pdb.set_trace()
        x = self.features(x)
        # print(x)
        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.embeddings(x)
        return x


def make_layers(mono=False, bn=False):
    layers = []
    if mono:
        in_channels = 1
    else:
        in_channels = 2

    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if bn:
                bn = nn.BatchNorm2d(v)
                layers += [conv2d, bn, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(cuda=True, mono=False, bn=False, num_classes=128):
    return VGG(make_layers(mono, bn), cuda=cuda, num_classes=num_classes)


def vggish(cuda=True, mono=False, bn=False, pretrain=False, num_classes=128):
    """
    VGGish is a PyTorch port of Tensorflow's VGGish architecture
    used to create embeddings for Audioset. It produces a 128-d
    embedding of a 960ms slice of audio.
    """
    model = _vgg(cuda=cuda, mono=mono, bn=bn, num_classes=num_classes)
    if pretrain and not bn:
        state_dict = hub.load_state_dict_from_url(VGGISH_WEIGHTS, progress=True)
        model.load_state_dict(state_dict, strict=True)
        tqdm.write('Loaded AudioSet Pretrained Weight!')
    # if pretrain:
    #     state_dict = hub.load_state_dict_from_url(VGGISH_WEIGHTS, progress=True)
    #     model.load_state_dict(state_dict, strict=False)
    #     tqdm.write('Loaded AudioSet Pretrained Weight!')

    if cuda:
        model = model.cuda()
    return model

if __name__ == "__main__":
    pass
