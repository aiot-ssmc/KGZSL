#  pip install node2vec
import itertools

import numpy
import pandas
from node2vec import Node2Vec

import utils
from apps import N
from apps.data_process import FMA_DIR
from apps.data_process.fma.kg_etc import FMA_KG_DIR, AllNodes, build_am

(N2V_DIR := FMA_KG_DIR / "node2vec").mkdir(exist_ok=True)


def get_graph(all_nodes: AllNodes):
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(map(lambda node: (node[0], {'name': node[1]}), all_nodes))

    def iter_all_edges():
        instrument_edges = list(build_am.iter_genre_instrument_conf(all_nodes))
        artist_edges = list(build_am.iter_genre_artist_conf(all_nodes))
        instrument_edges = map(lambda edge: (edge[0], edge[1], edge[2] * 5.), instrument_edges)
        return itertools.chain(instrument_edges, artist_edges)

    graph.add_weighted_edges_from(iter_all_edges())
    return graph


def get_model(graph, all_nodes, dimensions, p=1., q=1., ):
    walk_length = len(all_nodes) // len(all_nodes.genres)
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=1000,
                        p=p, q=q, workers=7)
    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model


def get_embeddings(model, all_nodes):
    embeddings = []
    for g_id, g_name in all_nodes.genres.items():
        node_id = all_nodes.genre_id_to_nid(g_id)
        node_vec = model.wv.get_vector(node_id)
        embeddings.append((g_id, g_name, node_vec))
    return pandas.DataFrame(embeddings, columns=['genre_id', 'genre_name', 'embedding'])


def vis_embeddings(embeddings: pandas.DataFrame):
    genre_info = utils.file.load_pkl(FMA_DIR / 'label_map' / "157_genres.pkl")
    names_list, vectors_list = [], []
    for g_id, g_name, node_vec in embeddings.values:
        g_top_id = genre_info.loc[g_id]['top_level']
        names_list.append(f"{g_top_id:04d}-{g_name}")
        vectors_list.append('\t'.join(map(str, node_vec)))
    utils.file.save_text('\n'.join(names_list), N2V_DIR / "names.tsv")
    utils.file.save_text('\n'.join(vectors_list), N2V_DIR / "vectors.tsv")


def append_embeddings_to_info(embeddings: pandas.DataFrame, file_path: str):
    genre_info = utils.file.load_pkl(file_path)
    genre_info[N.kg_embedding] = [numpy.zeros(128)] * len(genre_info)
    for g_id, g_name, node_vec in embeddings.values:
        assert g_name == genre_info.loc[g_id, N.name]
        genre_info.at[g_id, N.kg_embedding] = node_vec
    utils.file.save_to_pkl(genre_info, f"{file_path[:-4]}.kg_emb.pkl")


def main():
    all_nodes: AllNodes = utils.file.load_pkl(FMA_KG_DIR / "all_nodes.pkl")
    graph = get_graph(all_nodes)
    model = get_model(graph, all_nodes, dimensions=300, p=1., q=10.)
    model.save(str(N2V_DIR / "node2vec.model"))

    from gensim.models import Word2Vec
    model = Word2Vec.load(str(N2V_DIR / "node2vec.model"))
    embeddings = get_embeddings(model, all_nodes)
    vis_embeddings(embeddings)
    utils.file.save_to_pkl(embeddings, N2V_DIR / "embeddings.pkl")

    embeddings = utils.file.load_pkl(N2V_DIR / "embeddings.pkl")
    append_embeddings_to_info(embeddings, str(FMA_DIR / "label_map/157_genres.ismir_19-TRSRd.pkl"))


if __name__ == '__main__':
    main()
