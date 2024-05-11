import itertools
import random

import numpy

import utils
from dl import x_dataset
from dl.data import HParams, InfoFrame

from .gpu import args, fabric, tlog
from .. import N, SAMPLE_RATE, DATA_MIN_SECONDS

log = utils.log.get_logger()

dataloader_args = HParams(shuffle=True, pin_memory=True, num_workers=args.cpu_num)

hp = HParams(lr=1e-2, weight_decay=1e-4, max_lr_factor=1, grad_max_norm=10.,
             # init_div_factor=1., final_div_factor=1.,
             init_div_factor=25., final_div_factor=1e4,
             batch_size=8, total_epochs=50, steps_per_epoch=None,
             eval_batch_size=32,
             multi_cls_num=1, top_cls_num=1,
             pre_name="Mel", backbone_name="inception", head_name="simple",
             hidden_dim=128, feature_dim=1536, dropout=0.5,
             )

data_config = HParams(
    dataset="fma",
    need_top_genre=False,
    zsl=False,

    sample_method="u",  # "p" or "u" or "t"
    min_duration_seconds=0.5,
    max_duration_seconds=7,
    eval_duration_seconds=1.0,
    batch_size_multiplier=3,
)

dataset_dir = args.input / data_config.dataset


def re_split(old_conf):
    import sklearn.model_selection
    train_tid, val_tid = sklearn.model_selection.train_test_split(old_conf[N.training], test_size=0.1)
    _, test_tid = sklearn.model_selection.train_test_split(old_conf[N.validation], test_size=2000)
    return {
        N.training: train_tid,
        N.validation: val_tid,
        N.testing: test_tid,
    }


if data_config.dataset == "fma":
    if data_config.zsl:
        assert 'zsl' in args.config
        split_name = 'TRSRd'
        split_config = utils.file.load_pkl(dataset_dir / f"split/zsl-ismir_19-{split_name}.pkl")
        split_config = re_split(split_config)
        genre_info = InfoFrame.load(args.input / f"fma/label_map/157_genres.ismir_19-{split_name}.kg_emb.pkl")
        knowledge_graph_dir = args.input / "fma/knowledge_graph"
    else:
        assert 'zsl' not in args.config
        split_config = utils.file.load_pkl(dataset_dir / "split/classification-medium.pkl")
        genre_info = InfoFrame.load(args.input / "fma/label_map/157_genres.pkl")
        knowledge_graph_dir = args.input / "fma/knowledge_graph"

    top_genre_info = genre_info.filter_by_column(N.top_genre)
    data_config.need_top_genre = True
    hp.update(multi_cls_num=len(genre_info), top_cls_num=len(top_genre_info))

elif data_config.dataset == "mtg":
    assert "mtg" in args.config
    genre_info = InfoFrame.load(args.input / "mtg/label_map/genre.pkl")
    split_config = utils.file.load_pkl(dataset_dir / "split/autotagging_genre-split-0.pkl")
    hp.update(multi_cls_num=len(genre_info))


def get_dataloader(batch_size: int = 16):
    utils.output.dictionary(data_config, log.info)
    tlog.hyper_parameters(data_config, "data_config")

    all_dataset = x_dataset.XDataset.load_from_disk(dataset_dir / f'{data_config.dataset}_all_data')

    train_dataset = all_dataset.filter_by_ids(N.track_id, split_config[N.training])
    val_dataset = all_dataset.filter_by_ids(N.track_id, split_config[N.validation])
    test_dataset = all_dataset.filter_by_ids(N.track_id, split_config[N.testing])

    train_dataloader = (Datacollector(train_dataset, training=True)
                        .get_dataloader(batch_size=batch_size, **dataloader_args))
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = (Datacollector(val_dataset, training=False)
                      .get_dataloader(batch_size=batch_size, **dataloader_args))
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    test_dataloader = (Datacollector(test_dataset, training=False)
                       .get_dataloader(batch_size=batch_size, **dataloader_args))
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    if hp.steps_per_epoch is None:
        hp.steps_per_epoch = len(train_dataloader)
    else:
        assert hp.steps_per_epoch == len(train_dataloader)

    return train_dataloader, val_dataloader, test_dataloader


class Datacollector(x_dataset.TorchDataset):
    def __init__(self, dataset, training=True):
        super().__init__(dataset)
        self.training = training
        self.chunk_len = 1600
        self.sampler = Sampler(
            min_num=data_config.min_duration_seconds * SAMPLE_RATE // self.chunk_len,
            max_num=data_config.max_duration_seconds * SAMPLE_RATE // self.chunk_len,
            sample_method=data_config.sample_method)
        self.sample_len = lambda: self.sampler.sample_func() * self.chunk_len
        self.batch_size_multiplier = data_config.batch_size_multiplier
        self.eye4onehot = numpy.eye(len(genre_info), dtype=int)
        if not self.training:
            self.format_data = lambda x: x
            self.batch_size_multiplier = 1

    def get_dataloader(self, batch_size=1, **kwargs):
        if self.training:
            return super().get_dataloader(batch_size=batch_size, **kwargs)
        else:
            return super().get_dataloader(batch_size=1, **kwargs)

    def get_onehot_label(self, labels: list[int]):
        return self.eye4onehot[list(set(labels))].sum(axis=0)

    def format_data(self, audio_data):
        audio_len = self.sample_len()
        formatted_data = itertools.chain(*[utils.data.random_cut_list_array(audio_data, audio_len)
                                           for _ in range(self.batch_size_multiplier)])
        return list(formatted_data)

    def __getitems__(self, keys):
        items = self.dataset[keys]

        if data_config.need_top_genre:
            genre_top_label = [top_genre_info.id2label(gid) for gid in items[N.top_genre_id]]
        else:
            genre_top_label = [-1] * len(keys)

        genres_labels = [[genre_info.id2label(gid) for gid in gids] for gids in items[N.genre_ids]]
        genres_onehot_label = [self.get_onehot_label(labels) for labels in genres_labels]

        return {
            N.audio_data: self.format_data(items[N.audio_data]),
            N.multi_targets: genres_onehot_label * self.batch_size_multiplier,
            N.top_target: genre_top_label * self.batch_size_multiplier,
            N.track_id: items[N.track_id] * self.batch_size_multiplier,
        }


class Sampler:
    def __init__(self, min_num, max_num, sample_method):
        self.min_num = min_num
        self.max_num = max_num
        if sample_method == 'u':
            self.sample_func = self.u_sample_num
        elif sample_method == 't':
            self.sample_func = self.t_sample_num
        elif sample_method == 'p':
            self.sample_func = self.p_sample_num
        else:
            raise ValueError(f"sample method {sample_method} is not supported!")

    def u_sample_num(self):
        return random.randint(self.min_num, self.max_num)

    def t_sample_num(self):
        return int(numpy.random.triangular(self.min_num, self.min_num * 2, self.max_num + 1))

    def get_chunk_num_from_p(self, p, p_min, p_max):
        return self.min_num + int((p - p_min) / (p_max - p_min) * (self.max_num - self.min_num))

    def p_sample_num(self, alpha=2, max_pareto_num=25):  # probability<0.000128
        p = max_pareto_num + 1
        while p > max_pareto_num:
            p = numpy.random.pareto(alpha)
        return self.get_chunk_num_from_p(p, p_min=0, p_max=max_pareto_num)
