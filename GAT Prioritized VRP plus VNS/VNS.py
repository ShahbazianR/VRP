from scipy.spatial import distance
import numpy as np
import random
import time
import copy


def compute_distances(clients):
    distances = dict()
    for from_client in clients.keys():
        for to_client in clients.keys():
            distances[(from_client, to_client)] = distance.euclidean(clients[from_client], clients[to_client])
    return distances

def tour_distance(tour, dist_matrix):
    return sum(dist_matrix[(tour[i-1],tour[i])] for i in range(len(tour)))


def route_time_eval(tour, distance_matrix, C_TW, C_service, alpha=0.1, beta=0.2):
    print("Route Time Eval:", tour)
    route = copy.deepcopy(tour)[1:-1]
    route_service_time = 0
    route_tw_violation = 0
    total_time = distance_matrix[(tour[0],route[0])]

    ind = 0
    while ind < len(route)-1:
        i = route[ind]
        j = route[ind+1]
        dist_ij = distance_matrix[(i,j)]

        route_service_time += C_service[i]
        if total_time > C_TW[i][1]:
            route_tw_violation += beta*(total_time-C_TW[i][1])
        elif total_time < C_TW[i][0]:
            route_tw_violation += alpha*(C_TW[i][0]-total_time)
            total_time = C_TW[i][0]

        total_time += route_service_time
        total_time += dist_ij
        ind += 1

    # time_cost = route_service_time + route_tw_violation
    time_cost = route_tw_violation
    return time_cost


##############################
######### Operators ##########
##############################
##### 2 opt
def two_opt(route, i, j):
    tour = route[1:-1]
    new_tour = [route[0]] + tour[:i] + tour[i:j+1][::-1] + tour[j+1:] + [route[-1]]
    return new_tour


## Intra-route operators
## swapping two customers in the same route
def swap(route_, i, j):
    # print("Swap >>", route_, i, j)
    route = copy.deepcopy(route_)
    if len(route) > 1:
        route[i], route[j] = route[j], route[i]
    return route


## A customer is removed and reinserted in another position in the same route
def rand_reinsertion(route_, i, j):
    # print("Reinsertion >>", route_, i, j)
    route = copy.deepcopy(route_)[1:-1]
    items = [route_[i], route_[j]]
    if len(route) > 1:
        rand_ind_i = random.randint(0, len(route)-1)
        rand_ind_j = random.randint(0, len(route)-1)
        route.remove(items[0])
        route.remove(items[1])
        route.insert(rand_ind_i, items[0])
        route.insert(rand_ind_j, items[1])

    return [route_[0]]+ route + [route_[0]]



###################################
######### Main Functions ##########
###################################

## Roullette Method 
def roullette(success_dict):
    probs = dict()
    fitness_sum = sum([success_dict[key] for key in success_dict])+0.001
    previous_probability = 0
    for op in success_dict:
        fitness_op = success_dict[op]
        probs[op] = previous_probability + (fitness_op/fitness_sum)
        previous_probability = probs[op]

    random_number = random.random()
    selected_op = list(success_dict.keys())[0] ## Default value is the first one
    for key in success_dict.keys():
        if probs[key] > random_number:
            break; 
        selected_op = key
    return selected_op


def op_prob_update(probs, op, cost_S, cost_S_):
    probs[op] += (cost_S - cost_S_)/cost_S
    return probs


def local_search(tour, dist_matrix, operator):
    better_solution_found = True
    while better_solution_found:
        better_solution_found = False
        
        for i in range(1, len(tour) - 1):
            for j in range(i+1, len(tour)-1):
                if operator == two_opt and j-i == 1: continue

                new_tour = operator(tour, i, j)

                if tour_distance(new_tour, dist_matrix) < tour_distance(tour, dist_matrix):
                    tour = new_tour
                    better_solution_found = True

    return tour


def shaking(tour, k, operator=two_opt):
    if len(tour) <= 3:
        return tour
    new_tour = tour[:]
    for _ in range(k):
        i, j = sorted(random.sample(list(range(1, len(tour)-1)), 2))
        new_tour = operator(new_tour, i, j)
    return new_tour 


def vns(tour, dist_matrix, TW, ST, alpha, beta, k_max=100, dist_rate=1, time_rate=1):

    probs={two_opt:0.5, swap:0.5, rand_reinsertion:0.5}
    theta = 0.5

    k = 1
    total_exploration_time = 0
    total_exploitation_time = 0
    while k <= k_max:
        time_start = time.time()

        shake_operator = list(probs.keys())[random.randint(0, len(probs.keys())-1)]
        # print("Shake Operator:", shake_operator)
        k_tour = shaking(tour, k, shake_operator)
        time_end = time.time()

        total_exploration_time += time_end - time_start

        time_start = time.time()

        rand_num = random.random()

        if rand_num < theta:
            # print("Random, ", theta)
            operator = list(probs.keys())[random.randint(0, len(probs.keys())-1)]
        else:
            # print("Routlette")
            operator = roullette(probs)

        # print("Local Search Operator:", operator)
        new_tour = local_search(k_tour, dist_matrix, operator)
        time_end = time.time()
        total_exploitation_time += time_end - time_start

        cost_new_tour = dist_rate*tour_distance(new_tour, dist_matrix) + time_rate*route_time_eval(new_tour, dist_matrix, TW, ST, alpha, beta)
        cost_tour = dist_rate*tour_distance(tour, dist_matrix) + time_rate*route_time_eval(tour, dist_matrix, TW, ST, alpha, beta)

        if rand_num < theta:
            theta += (cost_tour - cost_new_tour)/cost_tour
            
        # print(">>>", theta, (cost_tour - cost_new_tour))
        # print("Operator:", operator)
        # print("Tour:", tour)
        # print("New Tour:", new_tour)
        # print("===================")

        if cost_new_tour < cost_tour:
            tour = new_tour
            probs = op_prob_update(probs, operator, cost_tour, cost_new_tour)
            k = 1
        else:
            k += 1
    return tour, total_exploration_time, total_exploitation_time
