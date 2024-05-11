import itertools
import utils
from apps.data_process import MTG_DIR
from apps.data_process.fma.kg_etc import AllNodes
from apps.data_process.mtg.kg_etc import genre_info, artists_info, instrument_info, MTG_KG_DIR, build_am

all_nodes: AllNodes = utils.file.load_pkl(MTG_KG_DIR / "all_nodes.pkl")

(N2V_DIR := MTG_KG_DIR / "node2vec").mkdir(exist_ok=True)

D = 300


def get_graph():
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(map(lambda node: (node[0], {'name': node[1]}), all_nodes))

    def iter_all_edges():
        genre_id_instrument_id_conf = utils.file.load_pkl(MTG_KG_DIR / "genre_id_instrument_id_conf.pkl")
        instrument_edges = list(build_am.iter_genre_conf(
            genre_id_instrument_id_conf, all_nodes.genre_id_to_nid, all_nodes.instrument_id_to_nid))
        genre_id_artist_id_conf = utils.file.load_pkl(MTG_KG_DIR / "genre_id_artist_id_conf.pkl")
        artist_edges = list(build_am.iter_genre_conf(
            genre_id_artist_id_conf, all_nodes.genre_id_to_nid, all_nodes.artist_id_to_nid))

        instrument_edges = map(lambda edge: (edge[0], edge[1], edge[2] * 5.), instrument_edges)
        return itertools.chain(instrument_edges, artist_edges)

    graph.add_weighted_edges_from(iter_all_edges())
    return graph


def vis_embeddings(embeddings):
    names_list, vectors_list = [], []
    for g_id, g_name, node_vec in embeddings.values:
        names_list.append(g_name)
        vectors_list.append('\t'.join(map(str, node_vec)))
    utils.file.save_text('\n'.join(names_list), N2V_DIR / "names.tsv")
    utils.file.save_text('\n'.join(vectors_list), N2V_DIR / "vectors.tsv")


def main():
    from apps.data_process.fma.kg_etc.embedding import get_model, get_embeddings
    # graph = get_graph()
    # model = get_model(graph, all_nodes, dimensions=300, p=1., q=10.)
    # model.save(str(N2V_DIR / "node2vec.model"))

    from gensim.models import Word2Vec
    model = Word2Vec.load(str(N2V_DIR / "node2vec.model"))
    embeddings = get_embeddings(model, all_nodes)
    vis_embeddings(embeddings)
    utils.file.save_to_pkl(embeddings, N2V_DIR / "embeddings.pkl")

    # from apps.data_process.fma.kg_etc.embedding import append_embeddings_to_info
    # embeddings = utils.file.load_pkl(N2V_DIR / "embeddings.pkl")
    # append_embeddings_to_info(embeddings, str(MTG_DIR / "label_map/***.pkl"))


if __name__ == '__main__':
    main()
