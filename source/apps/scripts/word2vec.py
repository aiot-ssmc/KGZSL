import pathlib
import numpy

import utils

log = utils.log.get_logger()


def load_model_google(vec_path):
    from gensim.models.keyedvectors import KeyedVectors
    return KeyedVectors.load_word2vec_format(vec_path, binary=True)


def load_model_text(text_data):
    from utils.log import progressbar
    model = {}
    for line in progressbar(text_data, desc="loading model"):
        if not line:
            continue
        word, *vec = line.split()
        if len(vec) % 100 != 0:
            log.warning(f"bad line: {word}, {vec[:10]}...{vec[-10:]}, len={len(vec)}")
            continue
        model[word] = numpy.array([float(x) for x in vec])
    return model


def load_model_glove(vec_path):
    return load_model_text(pathlib.Path(vec_path).read_text().rstrip().splitlines())


def load_model_fasttext(vec_path):
    text_data = pathlib.Path(vec_path).read_text().rstrip().splitlines()
    return load_model_text(text_data[1:])


words = ['blue', 'red', 'green',
         # 'black', 'white', 'orange', 'purple', 'pink',
         'blues', 'jazz', 'folk', 'pop',
         # 'ginger', 'soda', 'juice',
         'rock', 'classical', 'country',
         # 'experimental', 'instrumental', 'electronic', 'international', 'spoken'
         ]


def vis_and_save(vecs, file_name):
    from scipy.spatial.distance import pdist, squareform
    vec_dis = squareform(pdist(vecs, metric='euclidean'))
    utils.plot.data_matrix(vec_dis, title="distance matrix", x_ticks=words, y_ticks=words, fmt="1.2f",
                           show=True, out_path=f"{file_name}.dis.png")
    vec_dis = squareform(pdist(vecs, metric='cosine'))
    utils.plot.data_matrix(vec_dis, title="cosine distance matrix", x_ticks=words, y_ticks=words, fmt="1.2f",
                           show=True, out_path=f"{file_name}.cos_dis.png")

    vecs_2d = utils.data.reduce_dim(vecs, way="pca", dim=2)
    points_dict = {w: v for w, v in zip(words, vecs_2d)}
    utils.file.save_mat(f"{file_name}.mat", points_dict)
    fig, ax = utils.plot.gcf_a()
    for label, points in points_dict.items():
        ax.scatter(points[0], points[1], label=label)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.show()
    fig.savefig(f"{file_name}.png", dpi=100)


word2vec_model_dir = pathlib.Path(".../models/word2vec")


def main():
    log.info("model loading...")
    # model = load_model_google(word2vec_model_dir / "GoogleNews-vectors-negative300.bin")
    model = load_model_glove(word2vec_model_dir / "glove.840B.300d.txt")
    # model = load_model_fasttext(word2vec_model_dir / "fasttext_crawl-300d-2M-subword.vec")

    vecs = [model[w] for w in words]

    log.info("vecs loaded")
    vis_and_save(vecs, "word2vec/glove.840B.300d")
    log.info("done")


if __name__ == '__main__':
    main()
