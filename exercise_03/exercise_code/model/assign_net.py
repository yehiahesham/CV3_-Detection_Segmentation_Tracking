import torch
import torch.nn as nn
from torch.nn import functional as F
from exercise_code.model.distance_metrics import cosine_distance

class BipartiteNeuralMessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout=0.0):
        super().__init__()

        edge_in_dim = 2 * node_dim + 2 * edge_dim  # 2*edge_dim since we always concatenate initial edge features
        self.edge_mlp = nn.Sequential(
            *[
                nn.Linear(edge_in_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

        node_in_dim = node_dim + edge_dim
        self.node_mlp = nn.Sequential(
            *[
                nn.Linear(node_in_dim, node_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

    def edge_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Node-to-edge updates, as descibed in slide 71, lecture 5.
        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, 2 x edge_dim)
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)

        returns:
            updated_edge_feats = torch.Tensor with shape (|A|, |B|, edge_dim)
        """

        n_nodes_a, n_nodes_b, _ = edge_embeds.shape

        ########################################################################
        # TODO:                                                                #
        # update the input to the mlp updating the edge                        #
        # self.edge_in = ... # has shape (|A|, |B|, 2*node_dim + 2*edge_dim)   #
        # NOTE: Working with a bipartite graph allows us to vectorize all      #
        # operations in the formulas in a straightforward manner               #
        # (keep in mind that we store edge features in a matrix).              #
        # Given a node in A, it is connected to all nodes in B.                #
        ########################################################################
        # 2 * node_dim + 2 * edge_dim  # 2*edge_dim since we always concatenate initial edge features
        nodes_a_embeds_expanded = nodes_a_embeds.unsqueeze(1).expand(-1, n_nodes_b, -1)
        nodes_b_embeds_expanded = nodes_b_embeds.unsqueeze(0).expand(n_nodes_a ,-1, -1)

        # nodes_a_embeds_expanded=nodes_a_embeds.reshape(n_nodes_a, n_nodes_b)
        # nodes_b_embeds_expanded=nodes_b_embeds.reshape(n_nodes_a, n_nodes_b)
        self.edge_in = torch.cat( (nodes_a_embeds_expanded,nodes_b_embeds_expanded,edge_embeds) , dim=-1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        edge = self.edge_mlp(self.edge_in)
        return edge

    def node_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Edge-to-node updates, as descibed in slide 75, lecture 5.

        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, edge_dim)
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)

        Output:
            nodes_a: updated nodes_a_embeds, torch.Tensor with shape (|A|, node_dim)
            nodes_b: updated nodes_b_embeds, torch.Tensor with shape (|B|, node_dim)
        """

        ########################################################################
        # TODO:                                                                #
        # Prepare the features to generate a suitable input to the MLP network.#
        # You can have a look at the network definition to determine the shape.#
        #                                                                      #
        # For convenience, we provide you with the input shape to the MLP:     #
        # self.nodes_a_in = ... # shape (|A|, node_dim + edge_dim)             #
        # self.nodes_b_in = ... # shape (|B|, node_dim + edge_dim)             #
        #                                                                      #
        # NOTE: Use 'sum' as aggregation function                              #
        # NOTE: Working with a bipartite graph allows us to vectorize all      #
        # operations in the formulas in a straightforward manner               #
        # (keep in mind that we store edge features in a matrix).              #
        # Given a node in A, it is connected to all nodes in B.                #
        ########################################################################

        self.nodes_a_in = torch.cat((nodes_a_embeds,edge_embeds.sum(axis=1)), dim=-1)
        self.nodes_b_in = torch.cat((nodes_b_embeds,edge_embeds.sum(axis=0)), dim=-1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        nodes_a = self.node_mlp(self.nodes_a_in)
        nodes_b = self.node_mlp(self.nodes_b_in)

        return nodes_a, nodes_b

    def forward(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        edge_embeds_latent = self.edge_update(edge_embeds, nodes_a_embeds, nodes_b_embeds)
        nodes_a_latent, nodes_b_latent = self.node_update(edge_embeds_latent, nodes_a_embeds, nodes_b_embeds)

        return edge_embeds_latent, nodes_a_latent, nodes_b_latent


class AssignmentSimilarityNet(nn.Module):
    def __init__(self, reid_network, node_dim, edge_dim, reid_dim, edges_in_dim, num_steps, dropout=0.0):
        super().__init__()
        self.reid_network = reid_network
        self.graph_net = BipartiteNeuralMessagePassingLayer(node_dim=node_dim, edge_dim=edge_dim, dropout=dropout)
        self.num_steps = num_steps
        self.cnn_linear = nn.Linear(reid_dim, node_dim)
        self.edge_in_mlp = nn.Sequential(
            *[
                nn.Linear(edges_in_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        self.classifier = nn.Sequential(*[nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, 1)])

    def compute_motion_edge_feats(self, track_coords, current_coords, track_t, curr_t):
        """
        Computes initial edge feature tensor

        Args:
            track_coords: track's frame box coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates
                            has shape (num_boxes, 4)

            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)


        Returns:
            tensor with shape (num_trakcs, num_boxes, 5) containing pairwise
            position and time difference features
        """

        ########################################################################
        # TODO:                                                                #
        # Compute part of the initial edge feature vector.                     #
        # This part uses the information about the position and                #
        # time difference.                                                     #
        # In the previous exercises we used the IoU between the                #
        # bounding boxes.                                                      #
        # Here, you should implement the feature proposed in the task          #
        # description in the notebook.                                         #
        # It will later be concatenated with the reid_distance.                #
        #                                                                      #
        # NOTE: the format of the bounding boxes is (ltrb), meaning            #
        # (left edge, top edge, right edge, bottom edge). Remember the         #
        # orientation of the image coordinates.                                #
        #                                                                      #
        # NOTE 1: we recommend you to use box centers to compute distances     #
        # in the x and y coordinates.                                          #
        #                                                                      #
        ########################################################################

        
        track_t_h=track_coords[:,3]-track_coords[:,1]
        track_t_w=track_coords[:,2]-track_coords[:,0]
        
        current_coords_h=current_coords[:,3]-current_coords[:,1]
        current_coords_w=current_coords[:,2]-current_coords[:,0]

        x_dist= (current_coords[:,0]+current_coords_w//2).unsqueeze(0) - (track_coords[:,0]+track_t_w//2).unsqueeze(1)
        y_dist= (current_coords[:,1]+current_coords_h//2).unsqueeze(0) - (track_coords[:,1]+track_t_h//2).unsqueeze(1)
        
        feat1 = 2*(x_dist)/(track_t_h.unsqueeze(1)+current_coords_h.unsqueeze(0))
        feat2 = 2*(y_dist)/(track_t_h.unsqueeze(1)+current_coords_h.unsqueeze(0))
        feat3 = torch.log(track_t_h.unsqueeze(1)/current_coords_h.unsqueeze(0))
        feat4 = torch.log(track_t_w.unsqueeze(1)/current_coords_w.unsqueeze(0))
        feat5 = (curr_t.unsqueeze(0)-track_t.unsqueeze(1))
        
        feat1=feat1.unsqueeze(2)
        feat2=feat2.unsqueeze(2)
        feat3=feat3.unsqueeze(2)
        feat4=feat4.unsqueeze(2)
        feat5=feat5.unsqueeze(2)
        
        edge_feats=torch.cat([feat1,feat2,feat3,feat4,feat5],dim=2)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return edge_feats  # shape (num_tracks, num_boxes, 5)

    def forward(self, track_app, current_app, track_coords, current_coords, track_t, curr_t):
        """
        Args:
            track_app: track's reid embeddings, torch.Tensor with shape (num_tracks, 512)
            current_app: current frame detections' reid embeddings, torch.Tensor with shape (num_boxes, 512)
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)

            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)

        Returns:
            classified edges: torch.Tensor with shape (num_steps, num_tracks, num_boxes),
                             containing at entry (step, i, j) the unnormalized probability that track i and
                             detection j are a match, according to the classifier at the given neural message passing step
        """

        # Get initial edge embeddings
        edge_feats_app = cosine_distance(track_app, current_app)
        edge_feats_motion = self.compute_motion_edge_feats(track_coords, current_coords, track_t, curr_t)
        edge_feats = torch.cat((edge_feats_motion, edge_feats_app.unsqueeze(-1)), dim=-1)
        edge_embeds = self.edge_in_mlp(edge_feats)
        initial_edge_embeds = edge_embeds.clone()

        # Get initial node embeddings, reduce dimensionality from 512 to node_dim
        node_embeds_track = F.relu(self.cnn_linear(track_app))
        node_embeds_curr = F.relu(self.cnn_linear(current_app))

        classified_edges = []
        for _ in range(self.num_steps):
            # Concat current edge embeds with initial edge embeds, increasing the feature dimension
            edge_embeds = torch.cat((edge_embeds, initial_edge_embeds), dim=-1)
            # Override edge_embeds, node_embeds
            edge_embeds, node_embeds_track, node_embeds_curr = self.graph_net(
                edge_embeds=edge_embeds,
                nodes_a_embeds=node_embeds_track,
                nodes_b_embeds=node_embeds_curr
            )
            # Run the classifier on edge embeddings
            classified_edges.append(self.classifier(edge_embeds))
        classified_edges = torch.stack(classified_edges).squeeze(-1)
        similarity = torch.sigmoid(classified_edges)
        return similarity
