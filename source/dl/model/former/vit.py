import einops.layers.torch as torch_ops
import torch
from torch import nn

import utils
from . import transformer, pe

log = utils.log.get_logger()


class Image2Seq(nn.Module):
    """
    patch image to sequences.
    """

    def __init__(self, patch_size: tuple[int, int], embedding_dim: int = None, image_channels: int = 3):
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        patch_dim = self.patch_h * self.patch_w * image_channels
        dim = embedding_dim or patch_dim // 3
        self.net = nn.Sequential(
            torch_ops.Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=self.patch_h, pw=self.patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def pad(self, image: torch.Tensor):
        image_h, image_w = image.shape[-2:]
        padding_h = (self.patch_h - image_h % self.patch_h) % self.patch_h
        padding_w = (self.patch_w - image_w % self.patch_w) % self.patch_w
        if padding_h > 0 or padding_w > 0:
            image = nn.functional.pad(image, (0, padding_w, 0, padding_h))
        log.debug(f"padding image from ({image_h}, {image_w}) to ({image_h + padding_h}, {image_w + padding_w})")
        return image

    def forward(self, image):
        """
        Args:
            image: (batch_size, image_channels, image_h, image_w)
        Returns:
            (batch_size, seq_len, dim)
        """
        return self.net(self.pad(image))


class Vit(nn.Module):
    def __init__(self, patch_size: tuple[int, int], num_classes,
                 dim: int = None, depth: int = 6, head_num: int = 8,
                 pe_dropout: float = 0.1, dropout: float = 0.1, image_channels: int = 3,
                 max_seq_len=5000, add_cls_token=True, ):
        super().__init__()
        self.image_to_embedding = Image2Seq(patch_size, dim, image_channels)
        self.pe = pe.Sinusoidal(max_seq_len=max_seq_len, dim=dim, add_cls_token=add_cls_token,
                                dropout=pe_dropout, learnable=True)
        self.trans_cls = transformer.Classifier(dim=dim, num_classes=num_classes, depth=depth, head_num=head_num,
                                                dropout=dropout)

    def forward(self, img):
        x = self.image_to_embedding(img)
        x = self.pe(x)
        y = self.trans_cls(x)
        return y
