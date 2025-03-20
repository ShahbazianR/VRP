from scipy.spatial import distance
import numpy as np
import random
import time


def compute_distances(clients):
    distances = dict()
    for from_client in clients.keys():
        for to_client in clients.keys():
            distances[(from_client, to_client)] = distance.euclidean(clients[from_client], clients[to_client])
    return distances


def tour_distance(tour, dist_matrix):
    return sum([dist_matrix[(tour[i-1],tour[i])] for i in range(len(tour))])


def two_opt(route, i, j):
    tour = route[1:-1]
    new_tour = [route[0]] + tour[:i] + tour[i:j+1][::-1] + tour[j+1:] + [route[-1]]
    return new_tour


def local_search(tour, dist_matrix, operator):
    better_solution_found = True
    while better_solution_found:
        better_solution_found = False
        for i in range(1, len(tour) - 1):
            for j in range(i+1, len(tour)):
                if j-i == 1: continue
                new_tour = operator(tour, i, j)
                if  tour_distance(new_tour, dist_matrix) < tour_distance(tour, dist_matrix):
                    tour = new_tour
                    better_solution_found = True
    return tour

def shaking(tour, k):
    new_tour = tour[:]
    for _ in range(k):
        if len(tour)>3:
            i, j = sorted(random.sample(range(1, len(tour)-1), 2))
            new_tour = two_opt(new_tour, i, j)
    return new_tour


import copy
def route_time_eval(tour, distance_matrix, C_TW, C_service, alpha=0.1, beta=0.2):
    route = copy.deepcopy(tour)[1:-1]
    route_service_time = 0
    route_tw_violation = 0
    total_time = distance_matrix[(tour[0],route[0])]

    ind = 0
    while ind < len(route)-1:
        i = route[ind]
        j = route[ind+1]
        dist_ij = distance_matrix[(i,j)]

        if i in C_service.keys():
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


def routes_decompose(solution, depot):
    routes = []
    route = [solution[0]]
    for node_ind in range(1, len(solution)):
        node = solution[node_ind]
        if node == depot:
            if len(route):
                route.append(node)
                routes.append(route)
                route = []
            else:
                route.append(node)
        else:
            route.append(node)

    return routes


def routes_breaking(solution, demands, max_cap, depot):
    routes = routes_decompose(solution, depot)
    
    break_points = list()
    for route in routes:
        route_demand = 0
        if len(route) > 2:
            for i in route:
                if i in demands.keys():
                    route_demand += demands[i]
                    if route_demand > max_cap:
                        break_points.append(i)
                        route_demand = 0
                        
    solution_ = []
    for route in routes:
            solution_.extend(route)
    
    for item in break_points:
        item_index = solution.index(item)
        solution.insert(item_index-1, depot)
    
    return solution

def vns(tour, depot, demands, max_cap, dist_matrix, TW, ST, alpha=0.1, beta=0.2, dist_rate=1, time_rate=1, k_max=100, operator=two_opt):
    k = 1
    total_exploration_time = 0
    total_exploitation_time = 0

    start_time = time.time()
    while k <= k_max and (time.time()-start_time) < 1*15*60:
        print(f"VNS {k}, {(time.time()-start_time)}")
        time_start = time.time()
        k_tour = shaking(tour, k)
        time_end = time.time()
        total_exploration_time += time_end - time_start
        time_start = time.time()
        new_tour = local_search(k_tour, dist_matrix, operator)
        time_end = time.time()
        total_exploitation_time += time_end - time_start

        new_tour = routes_breaking(new_tour, demands, max_cap, depot)

        cost_new_tour = dist_rate*tour_distance(new_tour, dist_matrix) + time_rate*route_time_eval(new_tour, dist_matrix, TW, ST, alpha, beta)
        cost_tour = dist_rate*tour_distance(tour, dist_matrix) + time_rate*route_time_eval(tour, dist_matrix, TW, ST, alpha, beta)

        if cost_new_tour < cost_tour:
            tour = new_tour
            k = 1
        else:
            k += 1

    return tour, total_exploration_time, total_exploitation_time
