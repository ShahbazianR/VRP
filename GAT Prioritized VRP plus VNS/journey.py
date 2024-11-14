import pandas as pd
from utils_demo import f_get_results_plot_seaborn, f_get_results_plot_plotly
from utils import get_journey, read_from_pickle, create_vectors, load_tf_model
import tensorflow as tf
import numpy as np

data = pd.read_csv(r'./backup_results_VRP_20_2024-04-03.csv')
title = 'Graph_size = 20, Batch_size = #'
model_path = r'./model_checkpoint_epoch_1_VRP_20_2024-04-03.h5'
val_set_path = r'./Validation_dataset_VRP_20_2024-04-03.pkl'
validation_dataset = read_from_pickle(val_set_path)

# f_get_results_plot_seaborn(data_20_1024, title_20_1024)

embedding_dim=32
graph_size=20
n_encode_layers=3
n_agents = 2
n_depots = 3
alpha=0.1
beta=0.2

model = load_tf_model(model_path, embedding_dim=embedding_dim, graph_size=graph_size, num_depots=n_depots, n_encode_layers=n_encode_layers, N_Agents=n_agents, ALPHA=alpha, BETA=beta)

instances = []
for x in validation_dataset.batch(1):
    depots, customers, demand, time_windows = x
    cust_vectors, depot_vectors = create_vectors(customers, depots, demand, time_windows)
    cust_vectors = tf.convert_to_tensor(cust_vectors, dtype=tf.float64)
    depot_vectors = tf.convert_to_tensor(depot_vectors, dtype=tf.float64)
    instances.append((cust_vectors, demand, time_windows, depot_vectors, customers, depots))

# tour = [x for x in validation_dataset_20_1024.batch(1)][0]
tour = instances[0]
# print("Tour =>", tour)

## Inputs: custs, demands, time_windows, depots, coords_custs, coords_depot
cost, ll, pi = model(tour[0], tour[1], tour[2], tour[3], tour[4], tour[5], return_pi=True)

# print(cost, ll, pi)

cost = cost[0].numpy()[0]
ll = ll[0].numpy()[0]
pi = pi[0].numpy()[0]

print(f'Current cost=> cost:{cost}, cost_0:', round(cost[0],2))
print()
# print(f"====== Coordinates =====\nCustomers:\n{tour[4]}\n\nDeopts:\n{tour[5]}\n\nDemands:\n{tour[1]}\n\nTimesWindows:\n{tour[2]}")

routes_Agents = dict()
for agnt_id in range(model.n_agents):
    agent = model.agents[agnt_id]
    routes = list(agent.route.values())
    for r in routes:
        route = r
        if route[0] != 0:
            route.insert(0, 0)
        if route[-1]!= 0:
            route.append(0)

        list_of_paths = []
        cur_path = []
        for idx, node in enumerate(route):
            cur_path.append(node)
            if idx != 0 and node == 0:
                if cur_path[0] != 0:
                    cur_path.insert(0, 0)
                list_of_paths.append(cur_path)
                cur_path = []
        
        routes_Agents[agnt_id] = list_of_paths

print(f"Agents' routes =>", routes_Agents)

for agnt in list(routes_Agents.keys()):
    routes = routes_Agents[agnt]
    depot_id = int(agnt % n_depots)
    get_journey(tour, routes, depot_id, title=title)
