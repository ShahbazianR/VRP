import numpy as np
import tensorflow as tf
from tensorflow import keras 
from keras import layers, Model
from scipy.spatial import distance

from Environment import *
from Encoder import *
from Decoder import *
from utils import *
from VNS import compute_distances
import copy

## Assumption: We should have agents equal or greater then the number of depots

class GAT_RL(Model):
    def __init__(self, embedding_dim, capacity=1, battery_capacity=1, matching="fixed_order", 
                 n_depots=2, n_agents=2, n_encode_layers=2, num_heads=8, 
                 clip=10., alpha=0.1, beta=0.2):
        super().__init__()
        # attributes for MHA
        self.embedding_dim = embedding_dim
        self.n_attn_layers = n_encode_layers
        self.decode_type = matching

        self.n_heads = num_heads
        self.n_agents = n_agents
        self.n_depots = n_depots
        self.batch_size = None
        self.clip = clip
        self.alpha = alpha
        self.beta = beta

        self.visited = None

        self.battery_charge_capacity = battery_capacity

        self.capacity = capacity
        self.agents = [Agent for _ in range(self.n_agents)]
        self.current_agent_id = 0

        self.encoder = Encoder(input_dim=self.embedding_dim,
                                        num_heads=self.n_heads,
                                        num_layers=self.n_attn_layers)

        self.decoder = Decoder(embed_dim=self.embedding_dim, num_heads=self.n_heads, 
                               num_att_layers=self.n_attn_layers, clipping=self.clip)
        

    def call(self, custs, demands, time_windows, service_times, depots, coords_custs, coords_depot, return_pi=False):
        embeddings = self.encoder(custs, depots) ## Performing encoding
        self.batch_size = tf.shape(embeddings)[0]

        dc_masks = self.cluster_nodes(coords_custs, coords_depot)

        inputs = (custs, demands, time_windows, service_times)
        depot_ind = 0
        for agnt in range(self.n_agents):
            coord_depot = coords_depot[:,depot_ind,:][:, None, :]
            # print(f"Depot {depot_ind} => Assigned to the Agent {agnt}")
        
            coords = (coords_custs, coord_depot)
            self.agents[agnt] = Agent(inputs, coords, self.capacity, self.battery_charge_capacity, embeddings.shape[1], alpha=self.alpha, beta=self.beta)
            self.agents[agnt].depot_id = depot_ind

            depot_ind += 1
            if depot_ind >= self.n_depots:
                depot_ind = 0

        self.visited = tf.zeros((self.batch_size, 1, custs.shape[1]+1), dtype=tf.uint8) ## customers + depot for each batch 

        sequences = dict(zip(range(self.n_agents), [list()]*self.n_agents))
        outputs = dict(zip(range(self.n_agents), [list()]*self.n_agents))
    
        while not self.all_agents_finished(): 
            if self.decode_type == "fixed_order":
                agent = self.agents[self.current_agent_id]
                agent.visited = self.visited
                
                att_mask, _ = agent.get_att_mask()

                embeddings = self.encoder(custs, depots, att_mask)
                mask = agent.get_mask()

                dc_agent_mask = dc_masks
                selected_node, compatibility = self.decoder.fixed_order(embeddings, custs, dc_agent_mask, agent.depot_id, mask)

                # print("Selected Node:", selected_node, "Depot:", agent.depot_id)

                agent.step(selected_node.numpy())

                sequences[self.current_agent_id].append(selected_node)
                outputs[self.current_agent_id].append(compatibility[:, 0, :])

                self.visited = agent.visited
                self.current_agent_id = (self.current_agent_id+1) % self.n_agents

        costs = []
        lls = []
        pis = []
        for agnt_id in range(self.n_agents):
            agent = self.agents[agnt_id]
            _log_p, pi = tf.stack(outputs[agnt_id], 1), tf.cast(tf.stack(sequences[agnt_id], 1), tf.float32)
            pis.append(pi)
            coord_depot = agent.coords_depot
            cost = agent.get_costs(coord_depot, coords_custs, pi)
            ll = agent.get_log_likelihood(_log_p, pi)
            costs.append(cost)
            lls.append(ll)
        
        if return_pi:
            return costs, lls, pis

        return costs, lls

    def all_agents_finished(self,):
        return tf.reduce_all(tf.cast(self.visited[:, :, 1:], tf.bool))
    

    def cluster_nodes(self, custs, depots):

            def generating_clusters(C, D):
                clusters = dict()

                for d in D:
                    clusters[d] = list()

                Ds = dict()
                for c in C.keys():
                    min_depot = (None, np.inf)
                    for d in D.keys():
                        dist_cd = distance.euclidean(C[c], D[d])
                        if dist_cd < min_depot[1]:
                            min_depot = (d, dist_cd)
                    Ds[c] = min_depot

                for item in Ds:
                    depot = Ds[item][0]
                    cust = item
                    clusters[depot].append(cust)
                
                return clusters

            batch_size = custs.shape[0]
            custs_coords = custs.numpy()
            depts_coords = depots.numpy() 
            
            dc_mask = np.ones(shape=(batch_size, depots.shape[1], custs.shape[1]), dtype=bool)
            
            for batch in range(batch_size):
                c_coords_dict = dict(zip(range(len(custs_coords[batch])), [list(i) for i in custs_coords[batch]]))
                d_coords_dict = dict(zip(range(len(depts_coords[batch])), [list(i) for i in depts_coords[batch]]))
                clusters = generating_clusters(c_coords_dict, d_coords_dict)
                # print(batch, clusters)
                for depot in clusters.keys():
                    for index in clusters[depot]:
                        dc_mask[batch][depot][index] = False

            return tf.convert_to_tensor(dc_mask)


# if __name__ == "__main__":
#   bs = 2
#   c = 10
#   d = 3

#   custs = tf.random.normal([bs, c, 5]) ## assumingly, we have 10 customers
#   demands = tf.ones([bs, c]) ## assumingly, we have 10 demands equal to the number of customers
#   depots = tf.random.normal([bs, d, 5]) ## assumingly, we have 3 depots

#   timewindows = tf.random.normal([bs, c, 2])
#   service_times = tf.random.normal([bs, c, 1])

#   coords_custs = tf.random.normal([bs, c, 2])
#   coords_depot = tf.random.normal([bs, d, 2])

#   g_attn = GAT_RL(n_agents=3, n_depots=d, capacity= 1, battery_capacity=1, embedding_dim=32, num_heads=8, n_encode_layers=5)
#   g_attn(custs, demands, timewindows, service_times, depots, coords_custs, coords_depot)

# if __name__ == "__main__":
#     val_set_path = rf'./UValidation_dataset_VRP_20_2024-04-07.pkl'
#     validation_dataset = read_from_pickle(val_set_path)

#     instances = []
#     for x in validation_dataset.batch(1):
#         depots, customers, demand, time_windows, service_times = x
#         cust_vectors, depot_vectors = create_vectors(customers, depots, demand, time_windows, service_times)
#         cust_vectors = tf.convert_to_tensor(cust_vectors, dtype=tf.float64)
#         depot_vectors = tf.convert_to_tensor(depot_vectors, dtype=tf.float64)
#         instances.append((cust_vectors, demand, time_windows, service_times, depot_vectors, customers, depots))


#     tour = instances[0]
#     custs, demands, time_windows, service_times, depots, coords_custs, coords_depot = tour

#     total_coords = dict()
#     C = dict()
#     D = dict()

#     index = 0
#     n_depots = 3
#     for item in range(n_depots):
#         total_coords[index] = [coords_depot[0][item][0].numpy(), coords_depot[0][item][1].numpy()]
#         D[index] = [coords_depot[0][item][0].numpy(), coords_depot[0][item][1].numpy()]
#         index += 1

#     for item in range(len(coords_custs[0])):
#         total_coords[index] = [coords_custs[0][item][0].numpy(), coords_custs[0][item][1].numpy()]
#         C[index] = [coords_custs[0][item][0].numpy(), coords_custs[0][item][1].numpy()]
#         index += 1

#     # print("\n ======== Total Coordinates ========")
#     # print("Coordinates =>\n", total_coords, '\n')

#     distance_matrix = compute_distances(total_coords)
#     print("Distance Matrix Created")
#     print("================================================")

#     clusters = dict()
#     for d in D:
#         clusters[d] = list()

#     Ds = dict()
#     for c in C.keys():
#         min_depot = (None, np.inf)
#         for d in D.keys():
#             if distance_matrix[(c,d)] < min_depot[1]:
#                 min_depot = (d, distance_matrix[(c,d)])
#         Ds[c] = min_depot

#     for item in Ds:
#         depot = Ds[item][0]
#         cust = item
#         clusters[depot].append(cust)

#     print("Clusters:")
#     print(clusters)

#     print("================================================")

#     g_attn = GAT_RL(n_agents=3, n_depots=3, capacity= 1, battery_capacity=1, embedding_dim=32, num_heads=8, n_encode_layers=5)
#     g_attn(custs, demands, time_windows, service_times, depots, coords_custs, coords_depot)
