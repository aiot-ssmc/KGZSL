import itertools

import numpy
import torch
from torch import optim
import torch.utils.data

import dl
import utils
from apps import N, SAMPLE_RATE
from apps.config.data import data_config
from apps.config.nn import config
from apps.config.gpu import fabric, tlog

import apps.network
from apps.metrics import MultiMetric, TopMetric

hp = config.hp
log = utils.log.get_logger()
progressbar = tlog.progressbar

EVAL_LENGTH = int(SAMPLE_RATE * data_config.eval_duration_seconds)


class Model(dl.model.Lightning):
    def __init__(self):
        dl.model.Lightning.__init__(self, fabric)
        self.network = apps.network.NN()
        self.multi_metric = MultiMetric()
        self.top_metric = TopMetric() if data_config.need_top_genre else utils.struct.BlackHole()

    def __post_init__(self):
        # self.network.eval()
        # tlog.add_graph(self.network, torch.randn([3, SAMPLE_RATE]), use_strict_trace=False)
        # self.network.train()

        utils.output.dictionary(hp)
        tlog.hyper_parameters(hp)

        super().__post_init__()

    def configure_optimizers(self, lr=hp.lr, weight_decay=hp.weight_decay):
        return (
            optim.AdamW(
                itertools.chain(
                    self.network.preprocess.parameters(),
                    self.network.backbone.parameters(),
                    self.network.head_simple.parameters(),
                    self.network.head_extension.parameters(),
                ),
                lr=lr, weight_decay=weight_decay),
        )

    def configure_schedulers(self):
        schedulers = []
        for optimizer in self.optimizers:
            lr = dl.get_lr(optimizer)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=lr * hp.max_lr_factor,
                div_factor=hp.init_div_factor, final_div_factor=hp.final_div_factor,
                total_steps=hp.total_epochs)
            schedulers.append(scheduler)
        return schedulers

    def training_epoch(self, dataloader, ds_name="train"):
        """
        :param dataloader: progress_bar(torch.utils.data.DataLoader)
        :param ds_name: str
        :return: losses: list[float]
        """
        self.step_schedulers()
        total_losses = []
        for batch_input in dataloader:
            self.zero_grad_all_optimizers()
            nn_output = self.network(batch_input[N.audio_data])
            loss_dic = self.network.loss_func(nn_output, batch_input, ds_name)
            loss = loss_dic[N.total_loss]
            self.fabric.backward(loss)
            if hp.grad_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(parameters=self.network.parameters(),
                                               max_norm=hp.grad_max_norm)
            self.step_optimizers()
            total_losses.append(loss.item())
        loss_mean = numpy.mean(total_losses)
        tlog.variable(f"{ds_name}/loss_mean", loss_mean, prog_bar=True)
        current_lr = dl.get_lr(self.optimizers[-1])
        tlog.variable(f"{ds_name}/lr", current_lr)
        return loss_mean

    def eval_epoch(self, dataloader, ds_name="eval", enhance=False):
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
                    embeddings, multi_predicts, predicts = self.enhanced_network(batch_input[N.audio_data])
                else:
                    embeddings, multi_predicts, predicts = self.network(batch_input[N.audio_data])
                loss_dic = self.network.loss_func((embeddings, multi_predicts, predicts),
                                                  batch_input, ds_name)
                total_losses.append(loss_dic[N.total_loss].item())
                self.multi_metric.update(multi_predicts, batch_input[N.multi_targets])
                self.top_metric.update(predicts, batch_input[N.top_target])

                if i == 0:
                    self.multi_metric.export_sample(
                        audio_data=batch_input[N.audio_data][0],
                        multi_predict=multi_predicts[0],
                        multi_target=batch_input[N.multi_targets][0],
                        name=f"{ds_name}")

            tlog.variable(f"{ds_name}/loss_total_mean", numpy.mean(total_losses), prog_bar=True)
            top_metric_score = self.top_metric.output(name=ds_name)
            multi_metric_score = self.multi_metric.output(name=ds_name)
            return top_metric_score if data_config.need_top_genre else multi_metric_score

    def enhanced_network(self, batch_audio_data):
        assert len(batch_audio_data) == 1
        audio_data = batch_audio_data[0]
        if len(audio_data) > EVAL_LENGTH:
            audio_data = audio_data[:len(audio_data) // EVAL_LENGTH * EVAL_LENGTH]
            b_audio_data = audio_data.reshape([-1, EVAL_LENGTH])
        else:
            b_audio_data = audio_data.unsqueeze(0)
        results = [self.network(b_audio_data[i:i + hp.eval_batch_size]) for i in
                   range(0, len(b_audio_data), hp.eval_batch_size)]
        return self.enhance_result(map(torch.cat, zip(*results)))

    @staticmethod
    def enhance_result(results):
        embeddings, multi_predicts, predicts = results
        return (embeddings.mean(dim=0, keepdim=True),
                multi_predicts.mean(dim=0, keepdim=True),
                torch.log_softmax(predicts, dim=-1).sum(dim=0, keepdim=True),)
