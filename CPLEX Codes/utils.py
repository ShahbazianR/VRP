import tensorflow as tf
import pickle
from scipy.spatial import distance

## this function reads the instances from saved ones
def read_from_pickle(path, return_tf_data_set=True, num_samples=None):
    """Read dataset from file (pickle)
    """
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    objects = objects[0]
    if return_tf_data_set:
        depo, graphs, demand, time_windows, service_times = objects
        if num_samples is not None:
            return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand), list(time_windows), list(service_times))).take(num_samples)
        else:
            return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand), list(time_windows), list(service_times)))
    else:
        return objects
    

import numpy as np
def create_vectors(custs, depots, demands, time_windows):
    batch_size = custs.shape[0]
    n_custs = custs.shape[1]
    n_depots = depots.shape[1]
    
    custs_vectors = np.zeros(shape=(batch_size, n_custs, 5)) ## Each vector contains:location(2), demand(1), time_window(2)
    depot_vectors = np.zeros(shape=(batch_size, n_depots, 5)) 

    for batch in range(batch_size):
        for n_c in range(n_custs):
            location = custs[batch][n_c].numpy()
            demand = demands[batch][n_c].numpy()
            tw = time_windows[batch][n_c].numpy()
            custs_vectors[batch][n_c][0], custs_vectors[batch][n_c][1] = location
            custs_vectors[batch][n_c][2] = demand
            custs_vectors[batch][n_c][3], custs_vectors[batch][n_c][4] = tw

        for n_d in range(n_depots):
            location = depots[batch][n_d].numpy()
            demand = 0
            tw = [0, 1]
            depot_vectors[batch][n_d][0], depot_vectors[batch][n_d][1] = location
            depot_vectors[batch][n_d][2] = demand
            depot_vectors[batch][n_d][3], depot_vectors[batch][n_d][4] = tw

    return custs_vectors, depot_vectors

def compute_distances(clients):
    distances = dict()
    for from_client in clients.keys():
        for to_client in clients.keys():
            distances[(from_client, to_client)] = distance.euclidean(clients[from_client], clients[to_client])
    return distances