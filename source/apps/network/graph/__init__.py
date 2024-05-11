import torch
from torch import nn

from apps.data_process.fma.kg_etc import AllNodes
from apps.network.graph import ref


class GGNN(ref.GGNN):
    def __init__(self, all_nodes: AllNodes, adjacency_matrix, out_dim: int,
                 state_dim=4, annotation_dim=1, n_steps=5):
        self.all_nodes = all_nodes
        super().__init__(state_dim=state_dim, annotation_dim=annotation_dim,
                         n_edge_types=2, n_node=len(self.all_nodes), n_steps=n_steps)

        self.register_buffer("adjacency_matrix", torch.tensor(adjacency_matrix), persistent=False)
        self.padding_size = (self.n_node, self.state_dim - self.annotation_dim)
        self.out_emb = nn.Linear(self.n_node, out_dim)

        from apps.config.data import genre_info, top_genre_info
        assert len(genre_info) == len(self.all_nodes.genres)
        self.top_gids_order = top_genre_info.ids
        self.gids_order = genre_info.ids
        self.to(torch.float32)

    def forward(self, predict_labels: torch.Tensor, **kwargs):
        if predict_labels.shape[-1] == 16:
            annotation = self.get_annotation(predict_labels, with_order=self.top_gids_order)
        else:
            annotation = self.get_annotation(predict_labels, with_order=self.gids_order)

        return self.annotation_forward(annotation)

    def annotation_forward(self, annotation: torch.Tensor):
        padding = torch.zeros(len(annotation), *self.padding_size, dtype=annotation.dtype, device=annotation.device)
        prop_state = torch.cat((annotation, padding), 2)
        adjacency_matrix = self.adjacency_matrix.expand(len(annotation), -1, -1)
        annotation_out = super().forward(prop_state, annotation, adjacency_matrix)
        return self.out_emb(annotation_out)

    def get_annotation(self, predict_labels: torch.Tensor, with_order: list):
        annotation = torch.zeros((len(predict_labels), self.n_node),
                                 dtype=predict_labels.dtype, device=predict_labels.device)
        annotation[:, with_order] = predict_labels
        return annotation[:, :, None]
