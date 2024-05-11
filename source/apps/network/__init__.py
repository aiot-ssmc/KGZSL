import dl
from apps import N
from apps.config.gpu import tlog
from apps.network import classifier, pre_nn, inception, transformer, mixed


class NN(dl.model.Module):
    def __init__(self):
        from apps.config.nn import model_load_dir, config
        hp = config.hp
        super().__init__(model_dir=model_load_dir)

        self.preprocess = getattr(pre_nn, hp.pre_name)(hidden_dim=hp.hidden_dim)

        if hp.backbone_name == "inception":
            self.backbone = inception.Network(
                hidden_dim=hp.hidden_dim,
                dropout=hp.dropout,
                simplify=False,
            )
        elif hp.backbone_name == "transformer":
            self.backbone = transformer.Network(
                hidden_dim=hp.hidden_dim,
                dropout=hp.dropout,
                max_seq_len=4096,
            )
        elif hp.backbone_name == "mixed":
            self.backbone = mixed.Network(
                hidden_dim=hp.hidden_dim,
                dropout=hp.dropout[0],
                atten_dropout=hp.dropout[1],
            )
        else:
            raise ValueError(f"backbone_name: {hp.backbone_name} not support")

        self.head_simple, self.head_extension = (
            classifier.build_head(input_dim=hp.hidden_dim, output_feature_dim=hp.feature_dim,
                                  multi_cls_num=hp.multi_cls_num, top_cls_num=hp.top_cls_num,
                                  extension=hp.head_name))
        self.class_head = lambda hidden_x: self.head_extension(*self.head_simple(hidden_x))

        self.single_label_loss = dl.loss.Ignore(dl.loss.focal.Loss(gamma=2.0, weight=None, label_smoothing=0.05), -1)
        self.multi_label_loss = dl.loss.multi_label.CrossEntropyLoss()

    def forward(self, audio_data):
        self.eval()  # !
        # (batch, audios)
        x = self.preprocess(audio_data)
        # (batch, seq_n, hidden_dim)  0.1s*audio = 10*seq_n
        hidden_x = self.backbone(x)
        # (batch, hidden_dim)
        return self.class_head(hidden_x)

    def loss_func(self, nn_output, batch_input, ds_name):
        embeddings, multi_predicts, predicts, = nn_output
        loss_dic = dict(
            multi_label=self.multi_label_loss(multi_predicts, batch_input[N.multi_targets]) * 0.1,
            top_label=self.single_label_loss(predicts, batch_input[N.top_target])  # ! x10
        )
        return self.print_losses(loss_dic, ds_name)

    @staticmethod
    def print_losses(loss_dic: dict, ds_name=None):
        loss_dic[N.total_loss] = 0.0
        loss_dic[N.total_loss] = sum(loss_dic.values())
        if ds_name is not None:
            for loss_name, loss_value in loss_dic.items():
                tlog.variable(f"{ds_name}/loss_{loss_name}", loss_value.item(), prog_bar=True)
        return loss_dic
