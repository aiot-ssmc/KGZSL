from torch import Tensor, nn
from torchvision.models import inception
import utils

log = utils.log.get_logger()


class Network(nn.Module):
    def __init__(self, hidden_dim, dropout, simplify=False):
        super().__init__()
        self.inception = inception.Inception3(
            num_classes=hidden_dim, aux_logits=False, init_weights=True,
            transform_input=False, dropout=dropout)

        self.inception.Conv2d_1a_3x3 = (
            inception.BasicConv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        if simplify:
            self.simplify()

    def simplify(self):
        self.inception.Conv2d_2a_3x3 = nn.Identity()
        self.inception.Mixed_5d = nn.Identity()

        self.inception.Mixed_6b = nn.Identity()
        self.inception.Mixed_6c = nn.Identity()
        self.inception.Mixed_6d = nn.Identity()
        # self.inception.Mixed_6e = nn.Identity()

        self.inception.Mixed_7c = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        feature = self.inception(x[:, None, :, :])
        return feature
