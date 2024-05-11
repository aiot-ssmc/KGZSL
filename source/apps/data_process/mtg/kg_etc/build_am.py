import numpy

import utils
from apps.data_process.fma.kg_etc import AllNodes
from apps.data_process.mtg.kg_etc import genre_info, artists_info, instrument_info, MTG_KG_DIR


def build_all_nodes():
    genres_dict = {g_id: genre_info.id2name(g_id) for g_id in genre_info.ids}
    instruments_dict = {i_id: instrument_info.id2name(i_id) for i_id in instrument_info.ids}
    artists_dict = {a_id: artists_info.id2name(a_id) for a_id in artists_info.ids}
    all_nodes = AllNodes(genres=genres_dict, instruments=instruments_dict, artists=artists_dict)
    return all_nodes


def iter_genre_conf(genre_id_attr_id_conf, gid2nid, aid2nid):
    for genre_id, attr_conf in genre_id_attr_id_conf.iterrows():
        genre_nid = gid2nid(genre_id)
        for attr_id, conf in attr_conf.items():
            if conf != 0:
                attr_nid = aid2nid(attr_id)
                yield genre_nid, attr_nid, conf


def iter_all_edges(all_nodes):
    genre_id_artist_id_conf = utils.file.load_pkl(MTG_KG_DIR / "genre_id_artist_id_conf.pkl")
    genre_id_instrument_id_conf = utils.file.load_pkl(MTG_KG_DIR / "genre_id_instrument_id_conf.pkl")

    for genre_nid, instrument_nid, conf in iter_genre_conf(
            genre_id_instrument_id_conf, all_nodes.genre_id_to_nid, all_nodes.instrument_id_to_nid):
        yield genre_nid, 0, instrument_nid, conf
    for genre_nid, artist_nid, conf in iter_genre_conf(
            genre_id_artist_id_conf, all_nodes.genre_id_to_nid, all_nodes.artist_id_to_nid):
        yield genre_nid, 1, artist_nid, conf


def main():
    all_nodes = build_all_nodes()
    utils.file.save_to_pkl(all_nodes, MTG_KG_DIR / "all_nodes.pkl")
    from apps.data_process.fma.kg_etc.build_am import create_adjacency_matrix
    adjacency_matrix = create_adjacency_matrix(iter_all_edges(all_nodes), len(all_nodes), 2)
    numpy.savez_compressed(MTG_KG_DIR / "adjacency_matrix.npz", adjacency_matrix)
    print("done")


if __name__ == '__main__':
    main()
