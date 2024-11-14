import pandas as pd
from utils import read_from_pickle, create_vectors, load_tf_model
import tensorflow as tf
import numpy as np
from VNS import *

date = "2024-05-20"

data = pd.read_csv(rf'./backup_results_VRP_20_{date}.csv')
title = 'Graph_size = 20, Batch_size = #'
model_path = rf'./model_checkpoint_epoch_0_VRP_20_{date}.h5'
# val_set_path = rf'./Validation_dataset_VRP_20_{date}.pkl'
val_set_path = rf'./UValidation_dataset_VRP_20_2024-04-07.pkl'
# val_set_path = rf'./Validation_dataset_3.pkl'
validation_dataset = read_from_pickle(val_set_path)


## Time Window Penalty Coefficients
ALPHA = 0.1
BETA = 0.2
embedding_dim=32
graph_size=20
n_encode_layers=3
n_depots = 3
n_agents = n_depots
alpha=0.1
beta=0.2

instances = []
for x in validation_dataset.batch(1):
    depots, customers, demand, time_windows, service_times = x
    cust_vectors, depot_vectors = create_vectors(customers, depots, demand, time_windows, service_times)
    cust_vectors = tf.convert_to_tensor(cust_vectors, dtype=tf.float64)
    depot_vectors = tf.convert_to_tensor(depot_vectors, dtype=tf.float64)
    instances.append((cust_vectors, demand, time_windows, service_times, depot_vectors, customers, depots))


tour = instances[0]
custs, demands, time_windows, service_times, depots, coords_custs, coords_depot = tour

# print(f"====== Coordinates =====\nCustomers:\n{tour[4]}\n\nDepots:\n{tour[5]}\n\nDemands:\n{tour[1]}\n\nTimesWindows:\n{tour[2]}")


print(service_times)

total_coords = dict()
time_windows_customers = dict()
service_times_customers = dict()

C = dict()
D = dict()

index = 0
for item in range(n_depots):
    total_coords[index] = [coords_depot[0][item][0].numpy(), coords_depot[0][item][1].numpy()]
    D[index] = [coords_depot[0][item][0].numpy(), coords_depot[0][item][1].numpy()]
    index += 1

for item in range(len(coords_custs[0])):
    total_coords[index] = [coords_custs[0][item][0].numpy(), coords_custs[0][item][1].numpy()]
    C[index] = [coords_custs[0][item][0].numpy(), coords_custs[0][item][1].numpy()]
    time_windows_customers[index] = [time_windows[0][item][0].numpy(), time_windows[0][item][1].numpy()]
    service_times_customers[index] = service_times[0][item].numpy()
    index += 1

# print("\n ======== Total Coordinates ========")
# print("Coordinates =>\n", total_coords, '\n')

distance_matrix = compute_distances(total_coords)
print("Distance Matrix Created")
print("================================================")

clusters = dict()
for d in D:
    clusters[d] = list()

Ds = dict()
for c in C.keys():
    min_depot = (None, np.inf)
    for d in D.keys():
        if distance_matrix[(c,d)] < min_depot[1]:
            min_depot = (d, distance_matrix[(c,d)])
    Ds[c] = min_depot

for item in Ds:
    depot = Ds[item][0]
    cust = item
    clusters[depot].append(cust)

print("Clusters:")
print(clusters)

print("================================================")

model = load_tf_model(model_path, embedding_dim=embedding_dim, graph_size=graph_size, num_depots=n_depots, n_encode_layers=n_encode_layers, N_Agents=n_agents, ALPHA=alpha, BETA=beta)

print("Model Loaded")

cost, ll, pi = model(custs, demands, time_windows, service_times, depots, coords_custs, coords_depot, return_pi=True)

# print(cost,'\n', ll,'\n', pi)

cost = cost[0].numpy()[0]
ll = ll[0].numpy()[0]
pi = pi[0].numpy()[0]

print(f'Current cost=> cost:{cost}, cost_0:', round(cost[0],2))
print()

routes_Agents = dict()
for agnt_id in range(model.n_agents):
    agent = model.agents[agnt_id]
    routes = list(agent.route.values())
    print(agnt_id, "==>" ,routes)

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
                if len(cur_path) > 1:
                    list_of_paths.append(cur_path)
                    cur_path = []
        
        # print(list_of_paths, "\n===========")
        routes_Agents[agnt_id] = list_of_paths

print(f"Agents' routes =>", routes_Agents)

initial_solution = dict()

for key in routes_Agents.keys():
    initial_solution[key] = []
    for path in routes_Agents[key]:
        p = [key if x == 0 else x+(n_depots) for x in path]
        initial_solution[key].append(p)
        

# initial_solution = {0: [[0, 5, 12, 15, 16, 19, 0]], 1: [[1, 10, 22, 1]],
#                      2: [[2, 4,6, 7, 8, 11, 2], [2, 9, 13, 14, 17, 18, 2], [2, 20, 21, 2]]}

print(f"Initial Solution =>", initial_solution)

dist_rate = 1
time_rate = 1
TW = time_windows_customers
ST = service_times_customers
k_max = 200


final_solution = dict()
initial_distance = 0
final_distance = 0
for depot in initial_solution.keys():
    final_solution[depot] = []
    for route in initial_solution[depot]:
        if len(route) > 2:
            init_dist = tour_distance(route, distance_matrix)
            initial_distance += init_dist
            # print("Initial Distance =>", init_dist)
            best_tour, exploration_time, exploitation_time = vns(route, distance_matrix, TW, ST, alpha, beta, k_max, dist_rate=dist_rate, time_rate=time_rate)
            # print(best_tour, exploitation_time, exploration_time)
            fin_dist  = tour_distance(best_tour, distance_matrix)
            # print("Final Distance =>", fin_dist)
            final_distance += fin_dist
            final_solution[depot].append(best_tour)
            # print("===============================")

print("Initial Solution >>", initial_solution)
print("Initial Distance >>", initial_distance)
print()
print("Final Solution >>", final_solution)
print("Final Distance >>", final_distance)
