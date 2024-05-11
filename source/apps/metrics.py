import numpy
import torch
from torch import Tensor
import torchmetrics.classification as M

import dl
import utils
from apps import N
from apps.config.gpu import tlog
from dl.data import InfoFrame


class Metric(torch.nn.Module):
    @staticmethod
    def compute_metric(metric):
        result = metric.compute()
        metric.reset()
        return result


class MultiMetric(Metric):
    def __init__(self):
        super().__init__()
        from apps.config.data import genre_info
        self.genre_info = genre_info

        self.multi_cls_num = len(genre_info)
        self.precision_multi = M.MultilabelPrecision(num_labels=self.multi_cls_num, average="micro", ignore_index=-1)
        self.roc_auc_multi = M.MultilabelAUROC(num_labels=self.multi_cls_num, average="micro", ignore_index=-1)
        self.pr_auc_multi = M.MultilabelAveragePrecision(num_labels=self.multi_cls_num, average="micro",
                                                         ignore_index=-1)

    def update(self, predicts, targets):
        threshold = predicts.max(dim=-1, keepdim=True).values
        threshold[threshold > 0] = 0.
        self.precision_multi.update(predicts >= threshold, targets)
        self.roc_auc_multi.update(predicts, targets)
        self.pr_auc_multi.update(predicts, targets)

    def output(self, name):
        precision_multi = self.compute_metric(self.precision_multi)
        roc_auc_multi = self.compute_metric(self.roc_auc_multi)
        pr_auc_multi = self.compute_metric(self.pr_auc_multi)
        tlog.variable(f"{name}/precision_multi", precision_multi, prog_bar=True)
        tlog.variable(f"{name}/roc_auc_multi", roc_auc_multi, prog_bar=True)
        tlog.variable(f"{name}/pr_auc_multi", pr_auc_multi, prog_bar=True)
        return roc_auc_multi

    def export_sample(self, audio_data: Tensor, multi_predict, multi_target, name):
        multi_predict_name = [self.genre_info.label2name(i) for i, v in enumerate(multi_predict) if v > 0]
        multi_target_name = [self.genre_info.label2name(i) for i, v in enumerate(multi_target) if v > 0]
        tag = f"multi: p({multi_predict_name}) t({multi_target_name})"
        tlog.audio(name=name, audio=audio_data, tag=tag)


class TopMetric(Metric):
    def __init__(self):
        super().__init__()
        from apps.config.data import top_genre_info
        self.top_genre_info = top_genre_info

        self.top_cls_num = len(self.top_genre_info)
        self.acc_top = M.MulticlassAccuracy(num_classes=self.top_cls_num, average="micro", ignore_index=-1)
        self.recall_top = M.MulticlassRecall(num_classes=self.top_cls_num, average="micro", ignore_index=-1)
        self.precision_top = M.MulticlassPrecision(num_classes=self.top_cls_num, average="micro", ignore_index=-1)
        self.f1_top = M.MulticlassF1Score(num_classes=self.top_cls_num, average="micro", ignore_index=-1)
        self.pr_auc_top = M.MulticlassAveragePrecision(num_classes=self.top_cls_num, average="macro", ignore_index=-1)
        self.roc_auc_top = M.MulticlassAUROC(num_classes=self.top_cls_num, average="macro", ignore_index=-1)
        self.confusion_matrix = M.MulticlassConfusionMatrix(num_classes=self.top_cls_num, ignore_index=-1)

    def update(self, predicts, targets):
        self.acc_top.update(predicts, targets)
        self.recall_top.update(predicts, targets)
        self.precision_top.update(predicts, targets)
        self.f1_top.update(predicts, targets)
        self.pr_auc_top.update(predicts, targets)
        self.roc_auc_top.update(predicts, targets)
        self.confusion_matrix.update(predicts, targets)

    def output(self, name):
        acc_top = self.compute_metric(self.acc_top)
        recall_top = self.compute_metric(self.recall_top)
        precision_top = self.compute_metric(self.precision_top)
        f1_top = self.compute_metric(self.f1_top)
        pr_auc_top = self.compute_metric(self.pr_auc_top)
        roc_auc_top = self.compute_metric(self.roc_auc_top)

        confusion_matrix = self.compute_metric(self.confusion_matrix)
        confusion_matrix = dl.t2n(confusion_matrix)
        label_names = self.top_genre_info.names

        tlog.variable(f"{name}/acc_top", acc_top, prog_bar=True)
        tlog.variable(f"{name}/recall_top", recall_top, prog_bar=True)
        tlog.variable(f"{name}/precision_top", precision_top, prog_bar=True)
        tlog.variable(f"{name}/f1_top", f1_top, prog_bar=True)
        tlog.variable(f"{name}/pr_auc_top", pr_auc_top, prog_bar=True)
        tlog.variable(f"{name}/roc_auc_top", roc_auc_top, prog_bar=True)

        tlog.figure(f"{name}/confusion_matrix",
                    self.draw_cm(confusion_matrix, label_names, False, "Confusion Matrix"))
        tlog.figure(f"{name}/confusion_matrix_probability",
                    self.draw_cm(confusion_matrix, label_names, True, "Confusion Matrix Probability"))

        return acc_top

    @staticmethod
    def draw_cm(confusion_matrix, label_names, output_probability, title):
        fig, ax2p = utils.plot.gcf_a(dpi=300)
        utils.plot.confusion_matrix(confusion_matrix, ax2p=ax2p, label_names=label_names,
                                    output_probability=output_probability, title=title)
        return fig


class ZSLMetric(Metric):
    def __init__(self):
        super().__init__()
        self.distances = []
        self.labels = []

        from apps.config.data import genre_info
        self.genre_info: InfoFrame = genre_info
        self.seen_mask = self.genre_info[N.seen].to_numpy()
        self.label_description = [f"s-{label}" if seen else f"u-{label}"
                                  for label, seen in zip(self.genre_info.names, self.seen_mask)]

    def update(self, distances, targets, margin=0.0):
        if margin > 0.0:
            # set seen cls distance to inf if it is larger than MARGIN_THRESHOLD
            seen_cls_distances = distances[:, self.seen_mask]
            seen_cls_distances[(seen_cls_distances > margin)] = float('inf')
            distances[:, self.seen_mask] = seen_cls_distances
        distances, targets = dl.t2n(distances), dl.t2n(targets)
        self.labels.append(targets)
        self.distances.append(distances)

        threshold = numpy.sort(distances, axis=1)[:, 4]
        predicts = threshold[:, numpy.newaxis] - distances
        return torch.tensor(predicts)

    def output(self, name, top_k=5):

        labels = numpy.concatenate(self.labels)

        distances = numpy.concatenate(self.distances)

        acc_tops = self.accuracy(distances, labels, top_k=top_k)
        tlog.variable(f"{name}/acc_{top_k}", acc_tops[-1], prog_bar=True)
        tlog.variable(f"{name}/acc_2", acc_tops[1], prog_bar=False)
        tlog.variable(f"{name}/acc_1", acc_tops[0], prog_bar=True)

        # s_acc_tops = self.accuracy(distances, labels, top_k=top_k, mask=list(self.seen_mask))
        # tlog.variable(f"{name}/s_acc_{top_k}", s_acc_tops[-1], prog_bar=False)
        # tlog.variable(f"{name}/s_acc_2", s_acc_tops[1], prog_bar=False)
        # tlog.variable(f"{name}/s_acc_1", s_acc_tops[0], prog_bar=False)
        #
        # u_acc_tops = self.accuracy(distances, labels, top_k=top_k, mask=list(~self.seen_mask))
        # tlog.variable(f"{name}/u_acc_{top_k}", u_acc_tops[-1], prog_bar=False)
        # tlog.variable(f"{name}/u_acc_2", u_acc_tops[1], prog_bar=False)
        # tlog.variable(f"{name}/u_acc_1", u_acc_tops[0], prog_bar=False)

        # multi_label_distances = numpy.concatenate(self.multi_label_predicts)
        # multi_label_distances = 1 / (1 + numpy.exp(multi_label_distances))
        # log_results(multi_label_distances, labels, "mul")

        self.reset()
        return acc_tops[1]

    def accuracy(self, predict_dis: numpy.ndarray, labels: numpy.ndarray, top_k=1, mask=None):
        """
        Args:
            predict_dis: (batch_size, cls_num) # the smaller is the better
            labels: (batch_size, cls_num) # 1 is True
            top_k: int
            mask: list[bool] # False means infinitely large distance
        """
        if mask is not None:
            predict_dis = predict_dis.copy()
            predict_dis[:, ~numpy.array(mask)] = float('inf')
        batch_size = labels.shape[0]
        top_k = min(top_k, labels.shape[1])
        top_k_predicts = numpy.argsort(predict_dis, axis=1)[:, :top_k]
        top_k_labels = numpy.take_along_axis(labels, top_k_predicts, axis=1)
        top_k_correct_num = ((top_k_labels == 1).cumsum(axis=1) > 0).sum(axis=0)
        top_k_acc = top_k_correct_num / batch_size
        return top_k_acc

    def reset(self):
        self.distances = []
        self.labels = []

    def export_embeddings(self, embeddings: Tensor, multi_predicts: Tensor, multi_targets: Tensor, name="result"):
        tags = []
        for multi_predict, multi_target in zip(multi_predicts, multi_targets):
            multi_predict_name = [self.genre_info.label2name(i) for i, v in enumerate(multi_predict) if v > 0]
            multi_target_name = [self.genre_info.label2name(i) for i, v in enumerate(multi_target) if v > 0]
            tag = f"data-t{multi_target_name}-p{multi_predict_name}"
            tags.append(tag)
        embeddings = torch.cat([embeddings, self.labels_embedding], dim=0)
        tags = tags + self.label_description
        tlog.embeddings(name=name, embeddings=embeddings, labels=tags)
