import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(
            self,
            n_classes: int,
            latent_dim: int,
            channels: int,
            img_size: int,
    ):
        super().__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

class CNNGenerator(nn.Module):
    """
    DCGAN-style CNN generator with a flexible number of upsampling stages.

    Design goals:
      - Keep the class generic: supports grayscale/RGB and any power-of-two image size
        (starting from 4x4 and doubling at each stage), e.g., 32x32, 64x64, 128x128.
      - Preserve the conditional interface used in the MLP baseline:
        concatenate a one-hot-like label embedding with the latent vector (z) at the input.
      - Provide simple knobs: base_channels, use_batchnorm.
      - Safe defaults so it works out-of-the-box on MNIST (1x32x32).

    Args:
        n_classes (int): Number of classes (e.g., 10 for MNIST/CIFAR-10).
        latent_dim (int): Dimension of the input noise z.
        channels (int): Number of output channels (1=grayscale, 3=RGB).
        img_size (int): Final square image size. Must be power-of-two and >= 32 in Step 1.
        base_channels (int): Channel multiplier for feature maps (typical 64).
        use_batchnorm (bool): Use BatchNorm2d after each upsampling block (recommended).

    Forward signature:
        forward(noise: Tensor[B, latent_dim], labels: LongTensor[B]) -> Tensor[B, C, H, W]
    """
    def __init__(
        self,
        n_classes: int,
        latent_dim: int,
        channels: int,
        img_size: int,
        base_channels: int = 64,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        # Store basic shape info
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.channels = channels
        self.img_size = img_size

        # One-hot-like label embedding (same style as the MLP Generator)
        self.label_emb = nn.Embedding(n_classes, n_classes)

        # Compute how many times we need to upsample from 4x4 to img_size
        # Import locally to avoid touching module-level imports.
        import math
        assert img_size % 4 == 0 and (img_size & (img_size - 1) == 0), \
            "img_size must be a power of two and divisible by 4 (e.g., 32, 64, 128)."
        num_upsamples = int(math.log2(img_size // 4))

        # We start from a small 4x4 feature map with high channel count
        c0 = base_channels * (2 ** max(0, num_upsamples - 1))  # heuristic to keep channels reasonable

        in_dim = latent_dim + n_classes  # concat(z, onehot(y))
        self.fc = nn.Linear(in_dim, c0 * 4 * 4)

        # Build a list of upsampling blocks; each block doubles spatial dim
        blocks = []
        c_in = c0
        for _ in range(num_upsamples):
            c_out = max(base_channels, c_in // 2)
            blocks.append(self._upsample_block(c_in, c_out, use_batchnorm))
            c_in = c_out

        # Final "to image" conv: map to `channels`, use Tanh to get range [-1, 1]
        self.to_rgb = nn.Conv2d(c_in, channels, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential(*blocks)
        self.act = nn.Tanh()

        # (Optional) weight init can be added here if needed for stability.

    @staticmethod
    def _upsample_block(in_ch: int, out_ch: int, use_bn: bool) -> nn.Sequential:
        """
        A single upsampling block:
          - ConvTranspose2d (kernel=4, stride=2, pad=1): doubles H, W
          - BatchNorm2d (optional)
          - ReLU
        """
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Concatenate z and one-hot-like embedding of y
        # shapes: [B, latent_dim], [B, n_classes] -> [B, latent_dim + n_classes]
        gen_input = torch.cat((self.label_emb(labels), noise), dim=1)

        # Project to 4x4 feature map and reshape to NCHW
        x = self.fc(gen_input)                           # [B, c0*4*4]
        x = x.view(x.size(0), -1, 4, 4)                  # [B, c0, 4, 4]

        # Progressive upsampling to the target spatial size
        x = self.blocks(x)                               # [B, c_last, H, W] where H=W=img_size

        # Convert to image range [-1, 1]
        img = self.act(self.to_rgb(x))                   # [B, channels, img_size, img_size]
        return img

