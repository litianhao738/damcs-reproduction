# 模块说明：定义实验预留的视觉编码器与解码器网络。
import torch
import torch.nn as nn

# 变量说明（device）：Torch 张量和视觉模型执行计算的 CPU 或 GPU 设备。
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        # 变量说明（self.encoder）：把观察图像压缩为潜在特征的神经网络编码器。
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(12288, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
        )
        # 变量说明（self.decoder）：把潜在视觉特征还原为图像的神经网络解码器。
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 12288),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 12, 16)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 变量说明（x）：世界网格或图像中的横向坐标。
        x = self.encoder(x)
        x = self.decoder(x)
        return x