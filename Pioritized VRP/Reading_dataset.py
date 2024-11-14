from customer_class import Customer
from vehicle_class import Vehicle
import re

from configurations import config
battery_capacity, vehicle_capacity, vehicle_velocity, \
    vehicle_energy_decay, energy_consumption_per_distance = config()


def client_data_structure(nodes_data):
    customers = dict()
    customer_id = 0
    for row in range(len(nodes_data)):
        id, cx, cy = nodes_data['CUST_NO.'][row], float(nodes_data['XCOORD.'][row]), float(nodes_data['YCOORD.'][row])
        tw_start, tw_end = nodes_data['READY_TIME'][row], nodes_data['DUE_DATE'][row]
        quantity = float(nodes_data['DEMAND'][row])
        service_time =  float(nodes_data['SERVICE_TIME'][row])
        customers[customer_id] = Customer(customer_id, float(cx), float(cy), float(tw_start), float(tw_end),
                                          float(quantity), float(service_time))
        customer_id += 1
    return customers


def vehicle_depot_data_structure(nodes_data, depot_shift_index, vehicle_num):

    vehicles = dict()
    depots = dict()

    vehicle_id = 0
    depot_id = depot_shift_index

    for row in range(len(nodes_data)):
        ind, cx, cy = nodes_data['CUST_NO.'][row], float(nodes_data['XCOORD.'][row]), float(nodes_data['YCOORD.'][row])

        capacity = vehicle_capacity
        max_travel_time = nodes_data['DUE_DATE'][row]

        depots[depot_id] = {'dep_x':cx, 'dep_y':cy}
        depot_id += 1

    import random
    for i in range(vehicle_num):
        index = random.randint(min(list(depots.keys())), max(list(depots.keys())))
        coordinate = depots[index]
        cx = coordinate['dep_x']
        cy = coordinate['dep_y']
        vehicles[i] = Vehicle()
        vehicles[i].initiate(vehicle_id, float(cx), float(cy), depots, depots,
                                    float(capacity), float(max_travel_time), vehicle_energy_decay, battery_capacity, vehicle_velocity)
            

    return vehicles, depots


def creat_data_model(train_df, valid_df, test_df, depot_num, vehicle_num):
  depot_num_train = depot_num
  train_depots = train_df[0:depot_num_train]
  train_data = train_df[depot_num_train:]
  train_data = train_data.reset_index(drop=True)

  depot_num_valid = depot_num
  valid_depots = valid_df[0:depot_num_valid]
  valid_data = valid_df[depot_num_valid:]
  valid_data = valid_data.reset_index(drop=True)

  depot_num_test = depot_num
  test_depots = test_df[0:depot_num_test]
  test_data = test_df[depot_num_test:]
  test_data = test_data.reset_index(drop=True)

  train_clients_data = client_data_structure(train_data)
  valid_clients_data = client_data_structure(valid_data)
  test_clients_data = client_data_structure(test_data)

  train_depot_shift = len(train_clients_data)
  valid_depot_shift = len(valid_clients_data)
  test_depot_shift = len(test_clients_data)

  train_vehicles, train_depots = vehicle_depot_data_structure(train_depots, train_depot_shift, vehicle_num)
  valid_vehicles, valid_depots = vehicle_depot_data_structure(valid_depots, valid_depot_shift, vehicle_num)
  test_vehicles, test_depots = vehicle_depot_data_structure(test_depots, test_depot_shift, vehicle_num)

  train = {'clients': train_clients_data,
              'depots': train_depots,
              'vehicles': train_vehicles}

  valid = {'clients': valid_clients_data,
                'depots': valid_depots,
                'vehicles': valid_vehicles}

  test = {'clients': test_clients_data,
                'depots': test_depots,
                'vehicles': test_vehicles}

  return train, valid, test


def create_data_ind(dataframe, depot_num, vehicle_num):
  depots = dataframe[0:depot_num]
  data = dataframe[depot_num:]
  data = data.reset_index(drop=True)

  clients_data = client_data_structure(data)

  depot_shift = len(clients_data)

  data_vehicles, data_depots = vehicle_depot_data_structure(depots, depot_shift, vehicle_num)

  data = {'clients': clients_data,
              'depots': data_depots,
              'vehicles': data_vehicles}

  return data


import pandas as pd
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
    

import pandas as pd
def reading_vidal_ds(file_path):

    file = open(file_path, 'r')
    read_lines = file.readlines()
    type_number = re.split('\n', read_lines[0])[0]
    type_number = re.split(' ', type_number)

    vrp_type, n_vehicles, n_customers, days = type_number

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
