from vehicle_class import *
from customer_class import *
from PDQN_Network import *
from Reading_dataset import *
from scipy.spatial.distance import euclidean

import numpy as np
import copy
from tqdm import tqdm

def normal_dist(x, mean=0, sd=2):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

from configurations import config
battery_capacity, vehicle_capacity, vehicle_velocity, vehicle_energy_decay, energy_consumption_per_distance = config()


def energy_consumption(dist, current_energy, idle=False):
    energy = current_energy
    if idle:
        energy = current_energy * (1-vehicle_energy_decay)
    else:
        energy -= dist * (energy_consumption_per_distance)
    return energy


def RL_Agent(C, D, TW_C, Demands, Q, Clusters, T_Service, n_vehicles, n_episodes, n_time_steps, \
          Q_update_frequency = 10, Target_update_freq = 100, route_max_time=500, \
            alpha=0.1, beta=0.2):

    ## One Agent is assigned to each depot
    state_dim = 5
    action_dim = len(C)+len(D)

    envs = dict() ## one environment for each depot
    Agents = dict()

    cluster_demands = dict()
    cluster_TW = dict()
    cluster_coords = dict()
    cluster_service = dict()
    for depot in D.keys():
        cluster_demands[depot] = dict()
        cluster_TW[depot] = dict()
        cluster_coords[depot] = dict()
        cluster_service[depot] = dict()
        for client in Clusters[depot]:
            cluster_demands[depot][client] = Demands[client]
            cluster_TW[depot][client] = TW_C[client]
            cluster_coords[depot][client] = C[client]
            cluster_service[depot][client] = T_Service[client]

    Agents = DoubleDQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions = action_dim,
                eps_end=0.01, input_dims=state_dim, lr=0.003)
    Agents.Q_eval.exploration_proba = 0.5

    dv_assignment = dict()

    for v_num in range(n_vehicles):
        if v_num < len(D):
            if v_num not in dv_assignment.keys():
                dv_assignment[v_num] = []
        
            dv_assignment[v_num].append(v_num)
        else:
            dv_assignment[v_num-len(D)].append(v_num)

    print(dv_assignment)

    for episode in tqdm(list(range(n_episodes)), desc="PDQN Algorithm Episodes"):
        # print(f"\n======= episode {episode} started =======")
        cluster_demands = dict()
        cluster_TW = dict()
        cluster_coords = dict()
        for depot in D.keys():
            cluster_demands[depot] = dict()
            cluster_TW[depot] = dict()
            cluster_coords[depot] = dict()
            for client in Clusters[depot]:
                cluster_demands[depot][client] = Demands[client]
                cluster_TW[depot][client] = TW_C[client]
                cluster_coords[depot][client] = C[client]


        for depot in D.keys():
            envs[depot] = dict()
            for v in dv_assignment[depot]:
                envs[depot][v] = {
                        'done': False,
                        'total_routes': [],
                        'current_route': [],
                        'total_time': 0,
                        'total_distance': 0,
                        'current_route_time': 0,
                        'Q': copy.deepcopy(Q),
                        'idle': False,
                        'client_demands': copy.deepcopy(cluster_demands[depot]),
                        'client_TW': copy.deepcopy(cluster_TW[depot]),
                        'client_coords': copy.deepcopy(cluster_coords[depot]),
                        'client_service': copy.deepcopy(cluster_service[depot]),
                        'current_state_id': depot,
                        'current_state': np.zeros(shape=(1, 5))[0],
                        'next_state': np.zeros(shape=(1, 5))[0],
                        'unvisited_clients': copy.deepcopy(Clusters[depot]),
                        'energy': battery_capacity,
                        }
        

        global_time_step = 0
        update_steps = 0
        all_done = False
        while not all_done and global_time_step <= n_time_steps:
            for depot in D.keys():
                for v in dv_assignment[depot]:
                    if global_time_step < envs[depot][v]['total_time'] and envs[depot][v]['idle'] and not envs[depot][v]['done']:
                            ## The vehicle is idle
                            # print("IDLE", global_time_step)
                            reward = -1
                            envs[depot][v]['energy'] = energy_consumption(0, envs[depot][v]['energy'], idle=True)

                    elif global_time_step >= envs[depot][v]['total_time'] and not envs[depot][v]['done']:
                        remaining_demands = set()
                        for rmd_item in envs[depot][v]['unvisited_clients']:
                            remaining_demands.add(envs[depot][v]['client_demands'][rmd_item])

                        if 0 in remaining_demands:
                            remaining_demands.remove(0)

                        if len(remaining_demands):
                            min_demands = max(0, min(remaining_demands)) 
                            if envs[depot][v]['Q'] < min_demands or envs[depot][v]['current_route_time']>=route_max_time or envs[depot][v]['energy']<=0: 
                                ## Return to the depot
                                next_client = action = depot
                                reward = 0

                                state = np.zeros(shape=(1, 5))[0]
                                if envs[depot][v]['current_state_id'] in D.keys():
                                    state[0], state[1] = D[depot][0], D[depot][1]
                                elif envs[depot][v]['current_state_id'] in C.keys():
                                    client = envs[depot][v]['current_state_id']
                                    state[0], state[1] = C[client][0], C[client][1]
                                state[2], state[3] = D[depot][0], D[depot][1]
                                state[4] = reward                 

                                if len(envs[depot][v]['current_route']):
                                    envs[depot][v]['total_routes'].append(envs[depot][v]['current_route'])
                                    envs[depot][v]['current_route'] = []
                                envs[depot][v]['current_state'] = state
                                envs[depot][v]['current_state_id'] = depot
                                envs[depot][v]['Q'] = copy.deepcopy(Q)
                                envs[depot][v]['current_route_time'] = 0
                                envs[depot][v]['energy'] = battery_capacity

                            else:
                                current_client = envs[depot][v]['current_state_id']
                                current_state = envs[depot][v]['current_state']  
                                # print("Current State => ", current_state)
                                # print("Current state id =>", current_client)

                                envs[depot][v]['idle'] = False
                                reward = 0
                        
                                unseen_clients = copy.deepcopy(envs[depot][v]['unvisited_clients'])
                                # print(f"Unseen Clients for {depot} => ", unseen_clients)
                                for item in unseen_clients:
                                    if envs[depot][v]['Q'] < cluster_demands[depot][item]:
                                        unseen_clients.remove(item)
                                # print(f"Unseen Clients for {depot} => ", unseen_clients)

                                if len(unseen_clients):
                                    rand_num = np.random.uniform(0,1)
                                    if  rand_num < Agents.Q_eval.exploration_proba:
                                        # print("Random selection")
                                        rand_indx = np.random.randint(0,len(unseen_clients))
                                        next_client = unseen_clients[rand_indx]
                                    else:
                                        # print("Network Selection")
                                        actions = Agents.choose_action(current_state)
                                        q_values_detached = actions.cpu().detach().numpy()#[0]
                                        q_values = dict()
                                        for item in Clusters[depot]:
                                            q_values[item] = q_values_detached[item-len(D)]

                                        unseen_clients_qvalues = []
                                        for client in unseen_clients:
                                            unseen_clients_qvalues.append(q_values[client])
                                        
                                        selected_indx = np.argmax(unseen_clients_qvalues)
                                        next_client = unseen_clients[selected_indx]
                                    

                                    ############# Time Window Examination ##############
                                    tw_start = envs[depot][v]['client_TW'][next_client][0] 
                                    tw_end = envs[depot][v]['client_TW'][next_client][1]
                                    # print("Next client =>", next_client, envs[depot]['total_time'], tw_start, tw_end)

                                    if current_client in D.keys():
                                        current = (D[current_client][0], D[current_client][1])
                                    elif current_client in C.keys():
                                        current = (envs[depot][v]['client_coords'][current_client][0], envs[depot][v]['client_coords'][current_client][1])

                                    if tw_start <= envs[depot][v]['total_time'] <= tw_end: 
                                        envs[depot][v]['idle'] = False
                                        action = next_client

                                        next = (envs[depot][v]['client_coords'][next_client][0], envs[depot][v]['client_coords'][next_client][1])
                                        dist = euclidean(current, next)
                                        reward = 1/dist ##TODO: revise the reward function
                                        # reward = -dist 
                                        
                                        
                                        envs[depot][v]['total_time'] += dist
                                        envs[depot][v]['total_time'] += envs[depot][v]['client_service'][next_client]  ##the service time 
                                        
                                        envs[depot][v]['current_route_time'] += dist/vehicle_velocity
                                        envs[depot][v]['current_route_time'] += envs[depot][v]['client_service'][next_client]
                                        envs[depot][v]['total_distance'] += dist
                                        
                                        state = np.zeros(shape=(1, 5))[0]
                                        state[0], state[1] = current[0], current[1]
                                        state[2], state[3] = envs[depot][v]['client_coords'][next_client][0], envs[depot][v]['client_coords'][next_client][1]
                                        state[4] = reward  
                                        next_state = state

                                        Agents.store_transition(current_state, action, reward, next_state, envs[depot][v]['done'])
                                        envs[depot][v]['current_state'] = next_state
                                        envs[depot][v]['current_state_id'] = next_client
                                        envs[depot][v]['current_route'].append(next_client)
                                        envs[depot][v]['unvisited_clients'].remove(next_client)
                                        envs[depot][v]['Q'] -= envs[depot][v]['client_demands'][next_client]
                                        # print(envs[depot]['Q'], envs[depot]['client_demands'][next_client])

                                    elif envs[depot][v]['total_time'] < tw_start:
                                        envs[depot][v]['idle'] = True
                                        time_difference = tw_start - envs[depot][v]['total_time']
                                        action = next_client

                                        envs[depot][v]['current_route_time'] = tw_start ##shifting the time step to avoid wasting time
                                        
                                        next = (envs[depot][v]['client_coords'][next_client][0], envs[depot][v]['client_coords'][next_client][1])
                                        dist = euclidean(current, next)
                                        reward = 1/(dist + alpha*time_difference) ##TODO: revise the reward function
                                        # reward = -(dist + alpha*time_difference)


                                        envs[depot][v]['total_time'] += dist
                                        envs[depot][v]['total_time'] += envs[depot][v]['client_service'][next_client]  ##the service time 
                                        
                                        envs[depot][v]['current_route_time'] += dist/vehicle_velocity
                                        envs[depot][v]['current_route_time'] += envs[depot][v]['client_service'][next_client]

                                        envs[depot][v]['total_distance'] += dist
                                        
                                        state = np.zeros(shape=(1, 5))[0]
                                        state[0], state[1] = current[0], current[1]
                                        state[2], state[3] = envs[depot][v]['client_coords'][next_client][0], envs[depot][v]['client_coords'][next_client][1]
                                        state[4] = reward  
                                        next_state = state

                                        Agents.store_transition(current_state, action, reward, next_state, envs[depot][v]['done'])
                                        envs[depot][v]['current_state'] = next_state
                                        envs[depot][v]['current_state_id'] = next_client
                                        envs[depot][v]['current_route'].append(next_client)
                                        envs[depot][v]['unvisited_clients'].remove(next_client)
                                        envs[depot][v]['Q'] -= envs[depot][v]['client_demands'][next_client]
                                        # print(envs[depot]['Q'], envs[depot]['client_demands'][next_client])
                                    
                                    elif envs[depot][v]['total_time'] > tw_end: 
                                        ## ToDo: Implement this part
                                        envs[depot][v]['idle'] = False
                                        time_difference = envs[depot][v]['total_time'] - tw_end
                                        action = next_client

                                        next = (envs[depot][v]['client_coords'][next_client][0], envs[depot][v]['client_coords'][next_client][1])
                                        dist = euclidean(current, next)
                                        reward = 1/(dist + beta*time_difference) ##TODO: revise the reward function
                                        # reward = -(dist + beta*time_difference) 

                                        envs[depot][v]['total_time'] += dist
                                        envs[depot][v]['total_time'] += envs[depot][v]['client_service'][next_client]  ##the service time 
                                        
                                        envs[depot][v]['current_route_time'] += dist/vehicle_velocity
                                        envs[depot][v]['current_route_time'] += envs[depot][v]['client_service'][next_client]

                                        envs[depot][v]['total_distance'] += dist
                                        
                                        state = np.zeros(shape=(1, 5))[0]
                                        state[0], state[1] = current[0], current[1]
                                        state[2], state[3] = envs[depot][v]['client_coords'][next_client][0], envs[depot][v]['client_coords'][next_client][1]
                                        state[4] = reward  
                                        next_state = state

                                        Agents.store_transition(current_state, action, reward, next_state, envs[depot][v]['done'])
                                        envs[depot][v]['current_state'] = next_state
                                        envs[depot][v]['current_state_id'] = next_client
                                        envs[depot][v]['current_route'].append(next_client)
                                        envs[depot][v]['unvisited_clients'].remove(next_client)
                                        envs[depot][v]['Q'] -= envs[depot][v]['client_demands'][next_client]
                                        # print(envs[depot]['Q'], envs[depot]['client_demands'][next_client])

                                    envs[depot][v]['energy'] = energy_consumption(dist, envs[depot][v]['energy'], idle=False)
                                    
                                    remained_clients = copy.deepcopy(envs[depot][v]['unvisited_clients'])
                                    for v in dv_assignment[depot]:
                                        envs[depot][v]['unvisited_clients'] = copy.deepcopy(remained_clients)

                        if update_steps % Q_update_frequency == 0 and update_steps > 0:
                            loss = Agents.learn()
                            # print(f">> Training_phase: {loss}\n")

                        if update_steps % Target_update_freq == 0 and update_steps > 0:
                            Agents.target_Q = copy.deepcopy(Agents.Q_eval)


                        if not len(envs[depot][v]['unvisited_clients']):
                            envs[depot][v]['done'] = True
                        else:
                            envs[depot][v]['done'] = False

                        # print(envs[depot]['done'])
                        if envs[depot][v]['done']:
                            Agents.Q_eval.update_exploration_probability()
                            envs[depot][v]['total_routes'].append(envs[depot][v]['current_route'])
                            envs[depot][v]['current_route'] = []
                        
                
            global_time_step += 0.1
            update_steps += 1

            all_done = True
            for d in envs.keys():
                for v in envs[d]:
                    if not envs[d][v]['done']:
                        all_done = False
        
        # print(f"=========== episode {episode} finished ===========")
        total_dist = 0
        total_routes = dict()
        for e in envs.keys():
            total_routes[e] = dict()
            for v in envs[e]:
                # print(envs[e][v]['total_routes'])
                total_routes[e][v] = []
                total_routes[e][v] = envs[e][v]['total_routes']
                total_dist += envs[e][v]['total_distance']
        # print("Total distance:", total_dist)


    return Agents, total_routes