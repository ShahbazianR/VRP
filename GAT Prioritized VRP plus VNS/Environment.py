import tensorflow as tf
import numpy as np
from scipy.spatial.distance import euclidean


vehicle_energy_decay = 0.001
energy_consumption_per_distance = 0.02


class Agent():
    def __init__(self, input, coords, capacity:int, battery_capacity:int, dim:int, alpha, beta):
        
        # self.customers, self.demands, self.timewinows = input  ## ToDo: Add Time Windows
        self.customers, self.demands, self.time_windows, self.service_times = input
        # self.demands = tf.cast(self.demands, tf.float32)
        self.batch_size = self.customers.shape[0] # (batch_size, n_custs, 2)
        self.n_custs = self.customers.shape[1]

        self.vehicle_capacity = capacity
        self.depot_id = 0
        self.a_penalty = alpha
        self.b_penalty = beta

        ## TODO: Add battery charge
        self.total_charge = battery_capacity
        self.used_charge = tf.zeros((self.batch_size, 1), dtype=tf.float32)

        self.coords_custs, self.coords_depot = coords
        self.total_coords = tf.concat([self.coords_depot, self.coords_custs], axis=1)
        self.route = dict(zip(range(self.batch_size), [list()]*self.batch_size))

        self.batch_ids = tf.range(self.batch_size, dtype=tf.int64)[:, None]

        # State
        self.prev_a = tf.zeros((self.batch_size, 1), dtype=tf.float32)
        self.from_depot = self.prev_a == 0

        self.used_capacity = tf.zeros((self.batch_size, 1), dtype=tf.float32)
        self.visited = tf.zeros((self.batch_size, 1, self.n_custs+1), dtype=tf.uint8) ## customers + depot for each batch 

        # Constant tensors for scatter update (in step method)
        self.step_updates = tf.ones((self.batch_size, 1), dtype=tf.uint8)  
        self.scatter_zeros = tf.zeros((self.batch_size, 1), dtype=tf.int64) 

        self.i = tf.zeros(1, dtype=tf.int64)

        #############################################
        ## time arrival for customers and their time horizons;
        ## Each Batch => Each Customers => (a, t, b)
        self.tw_gap_a = np.zeros((self.batch_size, self.n_custs, 1))
        self.tw_gap_b = np.zeros((self.batch_size, self.n_custs, 1))
        self.current_time = tf.zeros((self.batch_size, 1), tf.float32)
        ## Assumingly, the current time is incremented by the distance taken at each step
        #############################################
    
    def product(self, a, b):
        return tf.einsum('ki,kj->kij', a, b)

    def get_att_mask(self):
        # Do not consider the assigned depot for being masked
        att_mask = tf.squeeze(tf.cast(self.visited, tf.float32), axis=-2)[:, 1:]  # [batch_size, 1, num_custs]
        
        # Number of nodes in new instance after masking
        total_nodes = self.n_custs + 1
        cur_num_nodes =  total_nodes - tf.reshape(tf.reduce_sum(att_mask, -1), (-1,1))  # [batch_size, 1]
        ## Add zero for the depot to each batch size
        att_mask = tf.concat((tf.zeros(shape=(att_mask.shape[0],1), dtype=tf.float32), att_mask), axis=-1)
        ones_mask = tf.ones_like(att_mask)
        
        # Create square attention mask from row-like mask
        att_mask = self.product(att_mask, ones_mask) + self.product(ones_mask, att_mask) - self.product(att_mask, att_mask)

        return tf.cast(att_mask, dtype=tf.bool), cur_num_nodes
    
    def all_finished(self):
        return tf.reduce_all(tf.cast(self.visited[:, :, 1:], tf.bool))
    
    def get_mask(self):
        """ Returns a mask (batch_size, 1, n_nodes) with available actions.
            Impossible nodes are masked.
        """
        # Exclude depot
        visited_loc = self.visited[:, :, 1:]

        # Mark nodes which exceed vehicle capacity
        exceeds_cap = (self.demands + self.used_capacity) > self.vehicle_capacity

        remaining_charge = self.used_charge < self.total_charge

        # We mask nodes that are already visited or have too much demand
        mask_loc = tf.cast(visited_loc, tf.bool) | exceeds_cap[:, None, :]

        # We can choose depot if 1) we are not in depot OR 2) all nodes are visited OR 3) We have run out of charge
        self.from_depot = tf.reshape(self.from_depot, (self.batch_size, 1))
        mask_depot = self.from_depot & (tf.reduce_sum(tf.cast(mask_loc == False, tf.int32), axis=-1) > 0) & remaining_charge
        #print("Masking Depot Conditions:", mask_depot, remaining_charge, self.from_depot)

        return tf.concat([mask_depot[:, :, None], mask_loc], axis=-1)
    
    def step(self, action):
        # Update current state
        selected = action[:, None]

        last_node = self.prev_a
        self.prev_a = selected
        self.from_depot = self.prev_a == 0

        # We have to shift indices by 1 since demand doesn't include depot
        # 0-index in demand corresponds to the FIRST node
        clipped_tensor = tf.clip_by_value(self.prev_a - 1, 0, self.n_custs - 1)
        clipped_tensor = tf.reshape(clipped_tensor, (clipped_tensor.shape[0], 1))
        selected_demand = tf.gather_nd(self.demands, tf.concat([self.batch_ids, clipped_tensor], axis=1))[:, None]  # (batch_size, 1)

        # We add current node capacity to used capacity and set it to zero if we return to the depot
        self.used_capacity = (self.used_capacity + selected_demand) * (1.0 - tf.cast(tf.reshape(self.from_depot, (self.batch_size, 1)), tf.float32))#[0]

        # Update visited nodes (set 1 to visited nodes)
        self.prev_a = tf.reshape(self.prev_a, (self.prev_a.shape[0], 1)) + 1 ## shifting the index one for the depot to fit in the visited nodes lists
        idx = tf.cast(tf.concat((self.batch_ids, self.scatter_zeros, self.prev_a), axis=-1), tf.int32)[:, None, :]  # (batch_size, 1, 3)
        self.visited = tf.tensor_scatter_nd_update(self.visited, idx, self.step_updates)  # (batch_size, 1, n_nodes)
        self.i = self.i + 1

        ## Evaluating Time Windows and the penalty for reaching late to a customer
        prevs_list = self.prev_a.numpy()-1
        for ind in range(len(prevs_list)):
            node = prevs_list[ind][0]
            self.route[ind] = self.route[ind] + [node]

        last_node = tf.cast(last_node, tf.int16).numpy()
        distances = dict()
        service_ts = dict()
        total_coords = self.total_coords.numpy()
        for batch_id in range(self.batch_size):
            last_coord = total_coords[batch_id][last_node[batch_id][0]]
            curr_coord = total_coords[batch_id][prevs_list[batch_id][0]]
            dist = euclidean(curr_coord, last_coord)
            distances[batch_id] = dist
            service_ts[batch_id] = self.service_times[batch_id][0]
        distances = tf.convert_to_tensor(list(distances.values()))
        distances = tf.reshape(distances, (self.batch_size, 1))
        self.current_time = tf.add(self.current_time, distances)

        idle_time = np.zeros((self.batch_size, 1))
        for batch_id in range(self.batch_size):
            curr_node = prevs_list[batch_id][0]
            curr_tw = self.time_windows[batch_id][curr_node].numpy()
            curr_time = self.current_time[batch_id].numpy()
            if curr_time < curr_tw[0]:
                self.tw_gap_a[batch_id][curr_node][0] = curr_tw[0]-curr_time
                idle_time[batch_id][0] = curr_tw[0]-curr_time
            if curr_time > curr_tw[1]:
                self.tw_gap_b[batch_id][curr_node][0] = curr_time-curr_tw[1]

        self.used_charge = (self.used_charge + energy_consumption_per_distance*distances + vehicle_energy_decay*idle_time) * (1.0 - tf.cast(tf.reshape(self.from_depot, (self.batch_size, 1)), tf.float32))

        services = tf.convert_to_tensor(list(service_ts.values()))
        self.current_time = tf.add(self.current_time, services)

    def get_costs(self, coord_depot, coord_custs, pi):
        # Place nodes with coordinates in order of decoder tour
        loc_with_depot = tf.concat([coord_depot, coord_custs], axis=1) # (batch_size, n_nodes, 2)
        d = tf.gather(loc_with_depot, tf.cast(pi, tf.int32), batch_dims=1)

        # Calculation of total distance
        # Note: first element of pi is not depot, but the first selected node in the path
        distance = (tf.reduce_sum(tf.norm(d[:, 1:] - d[:, :-1], ord=2, axis=2), axis=1)
                    + tf.norm(d[:, 0] - coord_depot, ord=2, axis=1) # Distance from depot to first selected node
                    + tf.norm(d[:, -1] - coord_depot, ord=2, axis=1))  # Distance from last selected node (!=0 for graph with longest path) to depot
        
        time_penalty = self.time_delay_evaluation().numpy()
        return distance + time_penalty
    
    def time_delay_evaluation(self,):
        alpha_value = self.a_penalty * tf.maximum(self.tw_gap_a, np.zeros(shape=self.tw_gap_a.shape)) 
        beta_value = self.b_penalty * tf.maximum(self.tw_gap_b, np.zeros(shape=self.tw_gap_b.shape))

        return tf.reduce_sum(alpha_value + beta_value)

    def get_log_likelihood(self, _log_p, a):
        # Get log_p corresponding to selected actions
        # log_p = tf.gather_nd(_log_p, tf.cast(tf.expand_dims(a, axis=-1), tf.int32), batch_dims=2)

        ## replacing -inf values with -20
        inf_values = tf.equal(_log_p, -np.inf)
        lp_masked = tf.where(inf_values, tf.ones(shape=inf_values.shape, dtype=_log_p.dtype)*(-200), _log_p)
        log_p = tf.gather_nd(lp_masked, tf.cast(tf.expand_dims(a, axis=-1), tf.int32), batch_dims=2)
        
        # Calculate log_likelihood
        return tf.reduce_sum(log_p,1)