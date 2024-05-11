import numpy
import umap

import utils.iter
from apps import N
from apps.config.gpu import tlog
from apps.config.nn import config, hp


def get_model():
    # assert config.config_args[0] == "transformer_zsl"
    from apps.model import Model
    return Model()


def generate_embedding():
    model = get_model()
    from apps.config.data import get_dataloader, top_genre_info
    _, val_loader, test_loader = get_dataloader(batch_size=hp.batch_size)

    # loader_to_use = utils.iter.sub(val_loader, 2000)
    loader_to_use = val_loader

    embedding_list = []
    label_list = []
    with model.evaluating():
        for batch_input in tlog.progressbar(loader_to_use, desc='generating embedding'):
            if batch_input[N.top_target][0] == -1:
                continue
            label_name = top_genre_info.label2name(batch_input[N.top_target][0].item())
            # if label_name in ["Classical", "Rock", "Experimental", "Electronic",
            #                   "Folk", "Pop", "Instrumental", "Jazz"]:
            embeddings, *_ = model.network(batch_input[N.audio_data])
            embedding_list.append(embeddings[0].detach().cpu().numpy())
            label_list.append(label_name)
    tlog.embeddings("embeddings", numpy.array(embedding_list), label_list)


def export():
    from apps.config import args
    data_dir = args.input.parent / ("output/scripts.feature_vis."
                                    "ggnn-debug-vis/40304111429"
                                    # "transformer_zsl-"
                                    # "PP-vis/40203175809"
                                    # "02-vis/40203175706"
                                    # "Rd-vis/40203175555"
                                    "/00000/embeddings/")
    embeddings = numpy.loadtxt(data_dir / "tensors.tsv")
    labels = utils.file.load_as_txt(data_dir / "metadata.tsv").rstrip().split('\n')
    # vis_f = utils.data.reduce_dim(embeddings, way="pca", dim=2)
    vis_f = umap.UMAP(metric="cosine",
                      n_neighbors=5, min_dist=0.99, n_epochs=400).fit_transform(embeddings)
    points_dict = {}
    for label in set(labels):
        points = vis_f[numpy.array(labels) == label]
        points_dict[label.replace(' / ', '').replace('-', '').replace(' ', '')] = points
    utils.file.save_mat(data_dir / "umap.mat", points_dict)
    points_dict = utils.file.load_mat(data_dir / "umap.mat")
    fig, ax = utils.plot.gcf_a()
    for label, points in points_dict.items():
        if label.startswith("_"):
            continue
        if label == 'OldTimeHistoric':
            ax.scatter(points[:, 0], points[:, 1], label=label, s=10)
        else:
            ax.scatter(points[:, 0], points[:, 1], label=label, s=1)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.show()
    print("done")


def main():
    export()


if __name__ == '__main__':
    main()
