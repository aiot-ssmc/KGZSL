import json
import random

import numpy
import pandas

import utils
from apps.data_process import FMA_DIR, SRC_OPEN_MIC_DIR
from apps.data_process.fma.kg_etc import AllNodes, FMA_KG_DIR
from dl.data import InfoFrame
from utils.log import progressbar

log = utils.log.get_logger()

genre_instrument_conf: pandas.DataFrame = utils.file.load_pkl(FMA_KG_DIR / "genre_id_inst_id_conf.pkl")
genre_artist_conf: pandas.DataFrame = utils.file.load_pkl(FMA_KG_DIR / "genre_id_artist_id_conf.pkl")


def create_adjacency_matrix(edges, n_node, n_edge):
    # (all start with 0)
    a = numpy.zeros([n_node, n_node * n_edge * 2])
    for src, e_type, tgt, conf in edges:
        a[tgt][e_type * n_node + src] = conf
        a[src][(e_type + n_edge) * n_node + tgt] = conf
    return a


def iter_genre_instrument_conf(all_nodes):
    for genre_id, instrument_conf in genre_instrument_conf.iterrows():
        genre_nid = all_nodes.genre_id_to_nid(genre_id)
        for instrument_id, conf in instrument_conf.items():
            if conf == conf:
                instrument_nid = all_nodes.instrument_id_to_nid(instrument_id)
                yield genre_nid, instrument_nid, conf


def iter_genre_artist_conf(all_nodes):
    for genre_id, artist_conf in genre_artist_conf.iterrows():
        genre_nid = all_nodes.genre_id_to_nid(genre_id)
        for artist_id, conf in artist_conf.items():
            if conf > 0:
                artist_nid = all_nodes.artist_id_to_nid(artist_id)
                yield genre_nid, artist_nid, conf


def iter_all_edges(all_nodes, instrument_random_rate=0., artist_random_rate=0.):
    random_genre_nid = lambda: random.randint(0, len(all_nodes.genres) - 1)
    random_instrument_nid = lambda: random.randint(len(all_nodes.genres),
                                                   len(all_nodes.genres) + len(all_nodes.instruments) - 1)
    random_artist_nid = lambda: random.randint(len(all_nodes.genres) + len(all_nodes.instruments),
                                               len(all_nodes) - 1)
    # for _ in range(1000):
    #     assert all_nodes.get_info(random_genre_nid())[0] == 'genre'
    #     assert all_nodes.get_info(random_instrument_nid())[0] == 'instrument'
    #     assert all_nodes.get_info(random_artist_nid())[0] == 'artist'
    for genre_nid, instrument_nid, conf in iter_genre_instrument_conf(all_nodes):
        if random.random() < instrument_random_rate:
            yield random_genre_nid(), 0, random_instrument_nid(), random.random()
        else:
            yield genre_nid, 0, instrument_nid, conf
    for genre_nid, artist_nid, conf in iter_genre_artist_conf(all_nodes):
        if random.random() < artist_random_rate:
            yield random_genre_nid(), 1, random_artist_nid(), random.random()
        else:
            yield genre_nid, 1, artist_nid, conf


def build_all_nodes():
    all_genre_ids = set(genre_instrument_conf.index)
    genre_info = InfoFrame.load(FMA_DIR / 'label_map' / "157_genres.pkl")
    genres_dict = {g_id: genre_info.id2name(g_id) for g_id in all_genre_ids}

    instruments_dict = json.load(open(SRC_OPEN_MIC_DIR / "class-map.json"))
    instruments_dict = {int(i_id): i_name for i_name, i_id in instruments_dict.items()}

    all_artist_ids = set(genre_artist_conf.columns)
    artists_info = InfoFrame.load(FMA_DIR / 'metadata' / "artists_info.pkl")
    artists_dict = {a_id: artists_info.id2name(a_id) for a_id in all_artist_ids}
    all_nodes = AllNodes(genres=genres_dict, instruments=instruments_dict, artists=artists_dict)
    return all_nodes


def main():
    # all_nodes = build_all_nodes()
    # utils.file.save_to_pkl(all_nodes, FMA_KG_DIR / "all_nodes.pkl")
    all_nodes = utils.file.load_pkl(FMA_KG_DIR / "all_nodes.pkl")
    adjacency_matrix = create_adjacency_matrix(
        progressbar(iter_all_edges(all_nodes, instrument_random_rate=0.0, artist_random_rate=0.0)),
        n_node=len(all_nodes), n_edge=2)
    numpy.savez_compressed(FMA_KG_DIR / "adjacency_matrix.npz", adjacency_matrix)
    print("done")


def get_balanced_am():
    all_nodes = utils.file.load_pkl(FMA_KG_DIR / "all_nodes.pkl")
    from apps.data_process.fma import load_all_tracks_info
    all_tracks_info = load_all_tracks_info()
    original_genres = all_tracks_info.loc[:, ('track', 'genres')]
    from collections import defaultdict
    genres_count = defaultdict(lambda: 0)
    for genres in original_genres:
        for gid in genres:
            genres_count[gid] += 1
    genres_weight = {k: max(genres_count.values()) / v for k, v in genres_count.items() if k in genre_artist_conf.index}
    genres_weight = pandas.Series(genres_weight)
    global genre_artist_conf
    genre_artist_conf = genre_artist_conf.mul(genres_weight, axis='index')
    adjacency_matrix = create_adjacency_matrix(
        progressbar(iter_all_edges(all_nodes, instrument_random_rate=0.0, artist_random_rate=0.0)),
        n_node=len(all_nodes), n_edge=2)
    numpy.savez_compressed(FMA_KG_DIR / "adjacency_matrix_balanced.npz", adjacency_matrix)
    print("done")


if __name__ == '__main__':
    main()
    # get_balanced_am()
