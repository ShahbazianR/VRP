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
    return sum(dist_matrix[(tour[i-1],tour[i])] for i in range(len(tour)))

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
                if tour_distance(new_tour, dist_matrix) < tour_distance(tour, dist_matrix):
                    tour = new_tour
                    better_solution_found = True
    return tour

def shaking(tour, k):
    if len(tour) <= 3:
        return tour
    new_tour = tour[:]
    for _ in range(k):
        i, j = sorted(random.sample(range(1, len(tour)-1), 2))
        new_tour = two_opt(new_tour, i, j)
    return new_tour

def vns(tour, dist_matrix, k_max=100, operator=two_opt):
    k = 1
    total_exploration_time = 0
    total_exploitation_time = 0
    while k <= k_max:
        time_start = time.time()
        k_tour = shaking(tour, k)
        time_end = time.time()
        total_exploration_time += time_end - time_start
        time_start = time.time()
        new_tour = local_search(k_tour, dist_matrix, operator)
        time_end = time.time()
        total_exploitation_time += time_end - time_start
        if tour_distance(new_tour, dist_matrix) < tour_distance(tour, dist_matrix):
            tour = new_tour
            k = 1
        else:
            k += 1

    return tour, total_exploration_time, total_exploitation_time
