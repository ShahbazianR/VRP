import pandas as pd
import re
def reading_cordeu_ds(file_path):

    file = open(file_path, 'r')
    read_lines = file.readlines()
    type_number = re.split('\n', read_lines[0])[0]
    type_number = re.split(' ', type_number)

    vrp_type, n_vehicles, n_customers, days = type_number
    print(vrp_type, n_vehicles, n_customers)

    max_duration = []
    max_load = []
    read_lines = read_lines[1:]
    for i in range(int(days)):
        line = read_lines[i]
        line = re.split('\n', line)[0]
        MD, MQ = re.split(' ', line)
        max_duration.append(int(MD))
        max_load.append(int(MQ))

    vehicles_dict = {'max_T': max_duration, 'max_load': max_load}

    depot_data = read_lines[-int(days):]
    data_lines = read_lines[int(days):-int(days)]
    data_df = []
    for line in data_lines:
        cust_line = re.split('\n', line)[0]
        cust_line = re.split(' ', line)
        cust_info = []
        for item in cust_line:
            if item != '':
                if '\n' in item:
                    item = re.split('\n', item)[0]
                cust_info.append(float(item))

        data_df.append(cust_info[0:5]+cust_info[-2:])
    
    data_df = pd.DataFrame(data_df, columns=['CUST_NO.', 'XCOORD.', 'YCOORD.', 'SERVICE_TIME', 'DEMAND', 'READY_TIME', 'DUE_DATE'])

    depot_df = []
    for line in depot_data:
        depot_line = re.split('\n', line)[0]
        depot_line = re.split(' ', line)
        depot_info = []
        for item in depot_line:
            if item != '':
                if '\n' in item:
                    item = re.split('\n', item)[0]
                depot_info.append(float(item))
        depot_df.append(depot_info[0:5]+depot_info[-2:])
    
    depot_df = pd.DataFrame(depot_df, columns=['CUST_NO.', 'XCOORD.', 'YCOORD.', 'SERVICE_TIME', 'DEMAND', 'READY_TIME', 'DUE_DATE'])

    vrp_types = {0:"VRP", 1:"PVRP", 2:"MDVRP", 3: "SDVRP", 4:"VRPTW", 5: "PVRPTW", 6: "MDVRPTW", 7:"SDVRPTW"}
    
    return data_df, depot_df, vehicles_dict, {'type':vrp_types[int(vrp_type)], 'n_vehicles':int(n_vehicles), 'n_customers':int(n_customers), 'days':int(days)}
    

from scipy.spatial import distance
import numpy as np
def compute_distances(clients):
    distances = []
    for from_client in clients:
        row = []
        for to_client in clients:
          row.append(distance.euclidean(from_client, to_client))
        distances.append(row)
    return np.array(distances)


def reading_vidal_ds(file_path):

    file = open(file_path, 'r')
    read_lines = file.readlines()
    type_number = re.split('\n', read_lines[0])[0]
    type_number = re.split(' ', type_number)

    vrp_type, n_vehicles, n_customers, days = type_number
    print(vrp_type, n_vehicles, n_customers)

    max_duration = []
    max_load = []
    read_lines = read_lines[1:]
    for i in range(int(days)):
        line = read_lines[i]
        line = re.split('\n', line)[0]
        MD, MQ = re.split(' ', line)
        max_duration.append(int(MD))
        max_load.append(int(MQ))

    vehicles_dict = {'max_T': max_duration, 'max_load': max_load}

    depot_data = read_lines[-int(days):]
    data_lines = read_lines[int(days):-int(days)]
    data_df = []
    for line in data_lines:
        cust_line = re.split('\n', line)[0]
        cust_line = re.split('\t', line)
        cust_info = []
        for item in cust_line:
            if item != '':
                if '\n' in item:
                    item = re.split('\n', item)[0]
                cust_info.append(float(item))

        data_df.append(cust_info[0:5]+cust_info[-2:])
    
    data_df = pd.DataFrame(data_df, columns=['CUST_NO.', 'XCOORD.', 'YCOORD.', 'SERVICE_TIME', 'DEMAND', 'READY_TIME', 'DUE_DATE'])

    depot_df = []
    for line in depot_data:
        depot_line = re.split('\n', line)[0]
        depot_line = re.split('\t', line)
        depot_info = []
        for item in depot_line:
            if item != '':
                if '\n' in item:
                    item = re.split('\n', item)[0]
                depot_info.append(float(item))
        depot_df.append(depot_info[0:5]+depot_info[-2:])
    
    depot_df = pd.DataFrame(depot_df, columns=['CUST_NO.', 'XCOORD.', 'YCOORD.', 'SERVICE_TIME', 'DEMAND', 'READY_TIME', 'DUE_DATE'])


    vrp_types = {0:"VRP", 1:"PVRP", 2:"MDVRP", 3: "SDVRP", 4:"VRPTW", 5: "PVRPTW", 6: "MDVRPTW", 7:"SDVRPTW"}
    
    
    return data_df, depot_df, vehicles_dict, {'type':vrp_types[int(vrp_type)], 'n_vehicles':int(n_vehicles), 'n_customers':int(n_customers), 'days':int(days)}
    

import tensorflow as tf
import pickle
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


def tour_distance(tour, dist_matrix):
    return sum(dist_matrix[(tour[i-1],tour[i])] for i in range(len(tour)))