import bidict

from apps.data_process import FMA_DIR

(FMA_KG_DIR := FMA_DIR / "knowledge_graph").mkdir(exist_ok=True)


class AllNodes:
    def __init__(self, genres: dict, instruments: dict, artists: dict):
        """
        (all start with 0)
        :param genres: {g_id: g_name}
        :param instruments: {i_id: i_name}
        :param artists: {a_id: a_name}
        """
        self.genres = genres
        self.instruments = instruments
        self.artists = artists
        nodes = (['g' + str(g_id) for g_id in genres.keys()] +
                 ['i' + str(i_id) for i_id in instruments.keys()] +
                 ['a' + str(a_id) for a_id in artists.keys()])
        self.nodes = bidict.bidict()
        for nid, node in enumerate(nodes):
            self.nodes[nid] = node

    def get_info(self, nid):
        return self.node_info(self.nodes[nid])

    def node_info(self, node):
        if node[0] == 'g':
            return 'genre', self.genres[int(node[1:])]
        elif node[0] == 'i':
            return 'instrument', self.instruments[int(node[1:])]
        elif node[0] == 'a':
            return 'artist', self.artists[int(node[1:])]
        else:
            raise ValueError(f'Unknown node type: {node}')

    def genre_id_to_nid(self, genre_id):
        return self.nodes.inv['g' + str(genre_id)]

    def instrument_id_to_nid(self, instrument_id):
        return self.nodes.inv['i' + str(instrument_id)]

    def artist_id_to_nid(self, artist_id):
        return self.nodes.inv['a' + str(artist_id)]

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes.items())
