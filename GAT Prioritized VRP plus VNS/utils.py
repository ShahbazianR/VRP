import pickle
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial import distance
from datetime import datetime
import GAT_RL_Model 
import numpy as np
import copy
import time


def save_to_pickle(filename, item):
    with open(filename, 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)

## generate random data instances while training the algorithm
def generate_data_onfly(num_samples=10000, graph_size=20, num_depots=2, max_time= 20):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }

    graph_size += 1

    depots, customers, demand = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, num_depots, 2)),
                            tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2)),
                            tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
                                        dtype=tf.int32), tf.float32)/tf.cast(CAPACITIES[graph_size-1], tf.float32))
    
    tw_a = tf.cast(tf.random.uniform(minval=0, maxval=max_time, shape=(num_samples, graph_size, 1), 
                                     dtype=tf.int32), tf.float32)/tf.cast(max_time, tf.float32)
    tw_b = tf.cast(tf.random.uniform(minval=0, maxval=max_time, shape=(num_samples, graph_size, 1), 
                                     dtype=tf.int32), tf.float32)/tf.cast(max_time, tf.float32)

    time_windows = tf.sort(tf.concat([tw_a, tw_b], axis=-1), axis=-1)

    service_times = tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 1), dtype=tf.float32)
    return tf.data.Dataset.from_tensor_slices((list(depots), list(customers), list(demand), list(time_windows), list(service_times)))


def create_data_on_disk(graph_size, num_samples, max_time=20, num_depots=2, is_save=True, filename=None, is_return=False, seed=1234):
    """Generate validation dataset (with SEED) and save
    """

    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }

    graph_size += 1

    depots, customers, demand = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, num_depots, 2)),
                            tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2)),
                            tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
                                        dtype=tf.int32), tf.float32)/tf.cast(CAPACITIES[graph_size-1], tf.float32))
    
    tw_a = tf.cast(tf.random.uniform(minval=0, maxval=max_time, shape=(num_samples, graph_size, 1), 
                                     dtype=tf.int32), tf.float32)/tf.cast(max_time, tf.float32)
    tw_b = tf.cast(tf.random.uniform(minval=0, maxval=max_time, shape=(num_samples, graph_size, 1), 
                                     dtype=tf.int32), tf.float32)/tf.cast(max_time, tf.float32)

    time_windows = tf.sort(tf.concat([tw_a, tw_b], axis=-1), axis=-1)

    # if is_save:
    #     save_to_pickle('Validation_dataset_{}.pkl'.format(filename), (depots, customers, demand, time_windows))

    # if is_return:
    #     return tf.data.Dataset.from_tensor_slices((list(depots), list(customers), list(demand), list(time_windows)))

    service_times = tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 1), dtype=tf.float32)

    if is_save:
        save_to_pickle(f'Validation_dataset_{filename}.pkl'.format(num_depots,filename), (depots, customers, demand, time_windows, service_times))

    if is_return:
        return tf.data.Dataset.from_tensor_slices((list(depots), list(customers), list(demand), list(time_windows), list(service_times)))



## this function creates the inital vectors for nodes in the graph, extracting their information from data instances
def create_vectors(custs, depots, demands, time_windows, service_times):
    batch_size = custs.shape[0]
    n_custs = custs.shape[1]
    n_depots = depots.shape[1]
    
    custs_vectors = np.zeros(shape=(batch_size, n_custs, 6)) ## Each vector contains:location(2), demand(1), time_window(2), service_time(1)
    depot_vectors = np.zeros(shape=(batch_size, n_depots, 6)) 

    for batch in range(batch_size):
        for n_c in range(n_custs):
            location = custs[batch][n_c].numpy()
            demand = demands[batch][n_c].numpy()
            tw = time_windows[batch][n_c].numpy()
            st = service_times[batch][n_c].numpy()
            custs_vectors[batch][n_c][0], custs_vectors[batch][n_c][1] = location
            custs_vectors[batch][n_c][2] = demand
            custs_vectors[batch][n_c][3], custs_vectors[batch][n_c][4] = tw
            custs_vectors[batch][n_c][5] = st

        for n_d in range(n_depots):
            location = depots[batch][n_d].numpy()
            demand = 0
            tw = [0, 1]
            depot_vectors[batch][n_d][0], depot_vectors[batch][n_d][1] = location
            depot_vectors[batch][n_d][2] = demand
            depot_vectors[batch][n_d][3], depot_vectors[batch][n_d][4] = tw
            depot_vectors[batch][n_d][5] = 0

    return custs_vectors, depot_vectors

## this function reads the instances from saved ones
# def read_from_pickle(path, return_tf_data_set=True, num_samples=None):
#     """Read dataset from file (pickle)
#     """
#     objects = []
#     with (open(path, "rb")) as openfile:
#         while True:
#             try:
#                 objects.append(pickle.load(openfile))
#             except EOFError:
#                 break
#     objects = objects[0]
#     if return_tf_data_set:
#         depo, graphs, demand, time_windows = objects
#         if num_samples is not None:
#             return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand), list(time_windows))).take(num_samples)
#         else:
#             return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand), list(time_windows)))
#     else:
#         return objects

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



## storing and visualizing the train and validation cost and loss results
def get_results(train_loss_results, train_cost_results, val_cost, save_results=True, filename=None, plots=True):

    epochs_num = len(train_loss_results)

    df_train = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                  'loss': train_loss_results,
                                  'cost': train_cost_results,
                                  })
    df_test = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                 'val_сost': val_cost})
    if save_results:
        df_train.to_excel('train_results_{}.xlsx'.format(filename), index=False)
        df_test.to_excel('test_results_{}.xlsx'.format(filename), index=False)

    if plots:
        plt.figure(figsize=(15, 9))
        ax = sns.lineplot(x='epochs', y='loss', data=df_train, color='salmon', label='train loss')
        ax2 = ax.twinx()
        sns.lineplot(x='epochs', y='cost', data=df_train, color='cornflowerblue', label='train cost', ax=ax2)
        sns.lineplot(x='epochs', y='val_сost', data=df_test, palette='darkblue', label='val cost').set(ylabel='cost')
        ax.legend(loc=(0.75, 0.90), ncol=1)
        ax2.legend(loc=(0.75, 0.95), ncol=2)
        ax.grid(axis='x')
        ax2.grid(True)
        plt.savefig('learning_curve_plot_{}.jpg'.format(filename))
        plt.show()


def get_journey(batch, total_routes, depot_id, title, ind_in_batch=0):
    """Plots journey of agents
    Args:
        batch: dataset of graphs
        pi: paths of agent obtained from model
        ind_in_batch: index of graph in batch to be plotted
    """

    # pi_ = []
    # for item in pi:
    #     pi_.append(int(item[0]))

    # pi_ = []
    # for route in total_routes:
    #     pi_.extend(route)

    # print("paths => ", pi_)

    # Unpack variables
    depo_coord = batch[-1][ind_in_batch].numpy()[depot_id]
    points_coords = batch[-2][ind_in_batch].numpy()
    demands = batch[1][ind_in_batch].numpy()
    node_labels = ['(' + str(x[0]+1) + ', ' + x[1] + ')' for x in enumerate(demands.round(2).astype(str))]

    # Concatenate depot and points
    depo_coord = np.reshape(depo_coord, (1, 2))
    full_coords = np.concatenate((depo_coord, points_coords))

    list_of_paths = total_routes
    print("List of paths", list_of_paths)

    list_of_path_traces = []
    for path_counter, path in enumerate(list_of_paths):
        coords = full_coords[[int(x) for x in path]]

        # Calculate length of each agent loop
        lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)

        list_of_path_traces.append(go.Scatter(x=coords[:, 0],
                                              y=coords[:, 1],
                                              mode="markers+lines",
                                              name=f"path_{path_counter}, length={total_length:.2f}",
                                              opacity=1.0))

    trace_points = go.Scatter(x=points_coords[:, 0],
                              y=points_coords[:, 1],
                              mode='markers+text',
                              name='destinations',
                              text=node_labels,
                              textposition='top center',
                              marker=dict(size=7),
                              opacity=1.0
                              )

    trace_depo = go.Scatter(x=[depo_coord[0][0]],
                            y=[depo_coord[0][1]],
                            text=['Depot'], textposition='bottom center',
                            mode='markers+text',
                            marker=dict(size=15),
                            name='depot'
                            )

    layout = go.Layout(title='<b>Example: {}</b>'.format(title),
                       xaxis=dict(title='X coordinate'),
                       yaxis=dict(title='Y coordinate'),
                       showlegend=True,
                       width=1000,
                       height=1000,
                       template="plotly_white"
                       )

    data = [trace_points, trace_depo] + list_of_path_traces
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def get_cur_time():
    """Returns local time as string
    """
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def load_tf_model(path, embedding_dim=128, graph_size=20, max_time=20, num_depots=2, n_encode_layers=2, N_Agents = 2, ALPHA=0.1, BETA=0.2):
    CAPACITIES = {10: 20.,
                  20: 30.,
                  50: 40.,
                  100: 50.
                  }

    num_samples = 2
    graph_size += 1
    depots, customers, demand = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, num_depots, 2)),
                            tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2)),
                            tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
                                        dtype=tf.int32), tf.float32)/tf.cast(CAPACITIES[graph_size-1], tf.float32))
    
    tw_a = tf.cast(tf.random.uniform(minval=0, maxval=max_time, shape=(num_samples, graph_size, 1), 
                                     dtype=tf.int32), tf.float32)/tf.cast(max_time, tf.float32)
    tw_b = tf.cast(tf.random.uniform(minval=0, maxval=max_time, shape=(num_samples, graph_size, 1), 
                                     dtype=tf.int32), tf.float32)/tf.cast(max_time, tf.float32)

    time_windows = tf.sort(tf.concat([tw_a, tw_b], axis=-1), axis=-1)
    service_times = tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 1), dtype=tf.float32)

    model_loaded = GAT_RL_Model.GAT_RL(embedding_dim, n_depots=num_depots , n_agents=N_Agents, n_encode_layers=n_encode_layers, num_heads=8, clip=10., alpha=ALPHA, beta=BETA)

    cust_vectors, depot_vectors = create_vectors(customers, depots, demand, time_windows, service_times)
    cust_vectors = tf.convert_to_tensor(cust_vectors, dtype=tf.float64)
    depot_vectors = tf.convert_to_tensor(depot_vectors, dtype=tf.float64)

    _, _ = model_loaded(cust_vectors, demand, time_windows, service_times, depot_vectors, customers, depots)

    model_loaded.load_weights(path)

    return model_loaded


#################################################
################ Test The File ##################
#################################################
# instances = generate_data_onfly(num_samples=2, graph_size=10)

# batch_size = 2
# for b in instances.batch(batch_size):
#     depots, customers, demands, time_windows = b
#     c, d = create_vectors(customers, depots, demands, time_windows)
#     print(c, d)
#     print("=====================")


