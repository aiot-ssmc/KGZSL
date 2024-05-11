import itertools

import numpy
import torch
from torch import nn

import apps.network
import apps.model
import dl
import utils
from apps import N
from apps.config.gpu import tlog
from apps.config.nn import config
from apps.metrics import ZSLMetric

hp = config.hp

EMB_NAME = N.kg_embedding

MARGIN = 0.1


class DD(nn.Module):
    def __init__(self):
        super().__init__()
        from apps.config.data import genre_info
        self.genre_info = genre_info
        labels_embedding = numpy.stack(self.genre_info[EMB_NAME])
        labels_embedding = utils.data.reduce_dim(labels_embedding, way="ICA", dim=hp.feature_dim)
        self.register_buffer("labels_embedding", torch.Tensor(labels_embedding), persistent=True)
        self.dis_method = self.cos_dis
        self.dis_delta = nn.Parameter(torch.ones([1, len(self.genre_info)]), requires_grad=True)
        self.loss = self.tri_loss
        # self.loss = lambda distances, targets: (1.0 * self.tri_loss(distances, targets)
        #                                         + 0.0 * self.hinge_loss(distances, targets))

    def forward(self, embeddings):
        distances = self.dis_func(embeddings, self.labels_embedding)
        return distances

    @staticmethod
    def cos_dis(features, target_features):
        """
        Args:
            features: (batch_size, cls_num, embedding_dim)
            target_features: (batch_size, cls_num, embedding_dim)
        Return:
            distances in (0, 2)
        """
        return 1 - nn.functional.cosine_similarity(features, target_features, dim=-1)

    def dis_func(self, features, target_features):
        """
        Args:
            features: (batch_size, embedding_dim)
            target_features: (cls_num, embedding_dim)
        Return:
            distances: (batch_size, cls_num) in (0, 2)
        """
        batch_size, embedding_dim = features.shape
        cls_num = target_features.shape[0]
        f_dim = (batch_size, cls_num, embedding_dim)
        target_features = target_features[None, :, :].expand(*f_dim)
        features = features[:, None, :].expand(*f_dim)
        distances = self.dis_method(features, target_features)
        new_distances = distances * self.dis_delta
        # new_distances[:, ~self.seen_mask] = distances[:, ~self.seen_mask]
        return new_distances

    def get_dis_delta(self):
        return list(zip(self.genre_info.names, dl.t2n(self.dis_delta[0])))

    @staticmethod
    def tri_loss(distances, targets):
        """
        Args:
            distances: (batch_size, multi_cls_num)
            targets: (batch_size, multi_cls_num)
        """
        distances = distances.clone()
        negative_samples = (targets != 1)

        # tri_loss
        distances[negative_samples] = torch.clamp(MARGIN - distances[negative_samples], min=0)
        loss = distances.sum() / targets.shape[0]
        return loss

    @staticmethod
    def hinge_loss(distances, targets):
        """
        Args:
            distances: (batch_size, multi_cls_num)
            targets: (batch_size, multi_cls_num)
        """
        distances = distances.clone()
        negative_samples = (targets != 1)
        loss = 0.0
        for i in range(distances.shape[0]):
            max_positive_distance = distances[i][~negative_samples[i]].max()
            min_negative_distance = distances[i][negative_samples[i]].min()
            if min_negative_distance - max_positive_distance < MARGIN / 2:
                loss += (max_positive_distance + MARGIN / 2) - min_negative_distance
        loss = loss / targets.shape[0]
        return loss


class NN(apps.network.NN):
    def __init__(self):
        super().__init__()
        self.zsl_dd = DD()

    def forward(self, audio_data):
        embeddings, multi_predicts, predicts = super().forward(audio_data)
        distances = self.zsl_dd(embeddings)
        return embeddings, distances, multi_predicts, predicts

    def loss_func(self, nn_output, batch_input, ds_name):
        embeddings, distances, multi_predicts, predicts = nn_output
        loss_dic = super().loss_func((embeddings, multi_predicts, predicts), batch_input, ds_name=None)
        loss_dic = {name: loss * 0.1 for name, loss in loss_dic.items()}  # !
        loss_dic["zsl"] = self.zsl_dd.loss(distances, batch_input[N.multi_targets])
        return self.print_losses(loss_dic, ds_name)


class Model(apps.model.Model):
    def __init__(self):
        super().__init__()
        self.network = NN()
        self.zsl_metric = ZSLMetric()

    def configure_optimizers(self, lr=hp.lr, weight_decay=hp.weight_decay):
        return (
            torch.optim.AdamW(
                itertools.chain(
                    self.network.preprocess.parameters(),
                    self.network.backbone.parameters(),
                    self.network.head_simple.parameters(),
                    self.network.head_extension.parameters(),
                ),
                lr=lr, weight_decay=weight_decay),
            torch.optim.AdamW(self.network.zsl_dd.parameters(), lr=lr * 0.1, weight_decay=0.0)
        )

    def eval_epoch(self, dataloader, ds_name="eval", enhance=True):
        """
        Args:
            dataloader: progress_bar(torch.utils.data.DataLoader)
            ds_name: str
            enhance: bool
        Return:
            losses: list[float]
        """
        total_losses = []
        with self.evaluating():
            for i, batch_input in enumerate(dataloader):
                if enhance:
                    embeddings, distances, multi_predicts, predicts = self.enhanced_network(batch_input[N.audio_data])
                else:
                    embeddings, distances, multi_predicts, predicts = self.network(batch_input[N.audio_data])
                loss_dic = self.network.loss_func((embeddings, distances, multi_predicts, predicts),
                                                  batch_input, ds_name)
                total_losses.append(loss_dic[N.total_loss].item())
                self.multi_metric.update(multi_predicts, batch_input[N.multi_targets])
                self.top_metric.update(predicts, batch_input[N.top_target])
                self.zsl_metric.update(distances, batch_input[N.multi_targets], margin=0.0)

                if i == 0:
                    self.multi_metric.export_sample(
                        audio_data=batch_input[N.audio_data][0],
                        multi_predict=multi_predicts[0],
                        multi_target=batch_input[N.multi_targets][0],
                        name=f"{ds_name}")

            tlog.variable(f"{ds_name}/loss_total_mean", numpy.mean(total_losses), prog_bar=True)
            top_metric_score = self.top_metric.output(name=ds_name)
            multi_metric_score = self.multi_metric.output(name=ds_name)
            zsl_metric_score = self.zsl_metric.output(name=ds_name)
            return zsl_metric_score

    @staticmethod
    def enhance_result(results):
        embeddings, distances, multi_predicts, predicts = results
        return (embeddings.mean(dim=0, keepdim=True),
                distances.mean(dim=0, keepdim=True),
                multi_predicts.mean(dim=0, keepdim=True),
                torch.log_softmax(predicts, dim=-1).sum(dim=0, keepdim=True),)
