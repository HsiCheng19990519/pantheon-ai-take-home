import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(
            self,
            n_classes: int,
            channels: int,
            img_size: int,
    ):
        super().__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

class CNNDiscriminator(nn.Module):
    """
    DCGAN-style CNN discriminator with a flexible number of downsampling stages.

    Conditioning strategy:
      - Embed label into a spatial "condition plane" of shape (1, H, W) and
        concatenate it with the input image along the channel dimension.
        This preserves the (img, label) interface without changing the step() logic.

    Args:
        n_classes (int): Number of classes.
        channels (int): Number of input image channels (1=grayscale, 3=RGB).
        img_size (int): Square image size (power-of-two, >= 32 in Step 1).
        base_channels (int): Feature width (typical 64).
        use_spectral_norm (bool): Wrap conv/linear layers with spectral_norm for stability.

    Forward signature:
        forward(img: Tensor[B, C, H, W], labels: LongTensor[B]) -> Tensor[B, 1]
        (Raw logits; DO NOT apply sigmoid when using hinge loss. LSGAN can use MSE directly.)
    """
    def __init__(
        self,
        n_classes: int,
        channels: int,
        img_size: int,
        base_channels: int = 64,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.img_size = img_size
        self.use_sn = use_spectral_norm

        # Label embedding -> spatial plane (1, H, W), then concat with image along C
        self.label_embedding = nn.Embedding(n_classes, img_size * img_size)

        # How many times to downsample from img_size to 4x4
        import math
        assert img_size % 4 == 0 and (img_size & (img_size - 1) == 0), \
            "img_size must be a power of two and divisible by 4."
        num_downsamples = int(math.log2(img_size // 4))

        def sn(layer: nn.Module) -> nn.Module:
            """Apply spectral norm if enabled."""
            return nn.utils.spectral_norm(layer) if self.use_sn else layer

        # First block: note the extra +1 channel from the condition plane
        c_in = channels + 1
        c_out = base_channels
        blocks = [
            nn.Sequential(
                sn(nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]

        # Intermediate downsampling blocks: double channels as we halve spatial dims
        for _ in range(num_downsamples - 1):
            c_next = min(c_out * 2, 512)  # cap to avoid exploding width
            blocks.append(
                nn.Sequential(
                    sn(nn.Conv2d(c_out, c_next, kernel_size=4, stride=2, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            c_out = c_next

        self.blocks = nn.Sequential(*blocks)
        self.head = sn(nn.Linear(c_out * 4 * 4, 1))  # final 4x4 -> 1 logit

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        # Build the (1, H, W) condition plane from label embedding
        cond = self.label_embedding(labels)             # [B, H*W]
        cond = cond.view(B, 1, H, W)                    # [B, 1, H, W]

        # Concatenate condition plane with the image along channel dim
        x = torch.cat([img, cond], dim=1)               # [B, C+1, H, W]

        # Progressive downsampling to 4x4
        x = self.blocks(x)                              # [B, c_out, 4, 4]

        # Flatten and map to a single logit
        x = x.view(B, -1)                               # [B, c_out*4*4]
        logit = self.head(x)                            # [B, 1]
        return logit
