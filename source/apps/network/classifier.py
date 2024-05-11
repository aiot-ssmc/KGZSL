import numpy
import torch
from torch import nn

import utils


def build_head(input_dim, output_feature_dim, multi_cls_num, top_cls_num, extension):
    simple_head = SimpleHead(input_dim, output_feature_dim, multi_cls_num, top_cls_num)
    if extension == 'simple':
        extension = Extension()
    elif extension == "complex":
        extension = ComplexExtension(output_feature_dim, multi_cls_num, top_cls_num)
    elif extension == "ggnn":
        extension = GGNNExtension(output_feature_dim, multi_cls_num, top_cls_num)
    else:
        raise ValueError(f"extension: {extension} not support")
    return simple_head, extension


class SimpleHead(nn.Module):
    def __init__(self, input_dim, output_feature_dim, multi_cls_num, top_cls_num):
        super().__init__()
        self.embedding_transform = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_feature_dim),
        )
        self.multi_label_nn = nn.Sequential(
            nn.Linear(output_feature_dim, output_feature_dim),
            nn.GELU(),
            nn.Linear(output_feature_dim, multi_cls_num),
        )
        self.target_label_nn = nn.Linear(multi_cls_num, top_cls_num)

    def forward(self, hidden_x):
        embedding_x = self.embedding_transform(hidden_x)
        multi_predicts = self.multi_label_nn(embedding_x)
        predicts = self.target_label_nn(multi_predicts)
        return embedding_x, multi_predicts, predicts


class Extension(nn.Module):
    def forward(self, embeddings, multi_predicts, predicts):
        return embeddings, multi_predicts, predicts


class ComplexExtension(Extension):
    def __init__(self, feature_dim, multi_cls_num, top_cls_num):
        super().__init__()

        self.multi_label_emb = nn.Linear(multi_cls_num, top_cls_num)
        self.target_label_emb = nn.Sequential(
            nn.Linear(feature_dim, top_cls_num),
            nn.GELU(),
        )

        self.target_label_nn = nn.Linear(top_cls_num * 2, top_cls_num)

    def forward(self, embeddings, multi_predicts, predicts):
        m_emb = self.multi_label_emb(multi_predicts)
        t_emb = self.target_label_emb(embeddings)
        predicts = self.target_label_nn(torch.cat([t_emb, m_emb], dim=-1))
        return embeddings, multi_predicts, predicts


class GGNNExtension(Extension):
    def __init__(self, feature_dim, multi_cls_num, top_cls_num):
        super().__init__()
        from apps.network import graph
        from apps.config.data import knowledge_graph_dir
        all_nodes = utils.file.load_pkl(knowledge_graph_dir / "all_nodes.pkl")
        # !
        adjacency_matrix = numpy.load(knowledge_graph_dir / "adjacency_matrix.npz", allow_pickle=True)['arr_0']
        self.ggnn = graph.GGNN(all_nodes, adjacency_matrix, out_dim=feature_dim, n_steps=5)
        import torchvision
        self.se = torchvision.ops.SqueezeExcitation(2, 2)
        self.head = SimpleHead(feature_dim * 2, feature_dim, multi_cls_num, top_cls_num)

    def forward(self, embeddings, multi_predicts, predicts):
        embeddings, multi_predicts, predicts = embeddings.detach(), multi_predicts.detach(), predicts.detach()
        self.train()  # !
        # kg_emb = self.ggnn(multi_predicts)
        kg_emb = self.ggnn(torch.softmax(predicts, dim=-1))  # !
        embeddings = torch.stack([embeddings, kg_emb], dim=1)
        embeddings = self.se(embeddings[..., None])  # ! w/o se
        return self.head(embeddings.view(embeddings.shape[0], -1))
