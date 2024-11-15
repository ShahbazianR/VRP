from scipy.spatial import distance
from sklearn import tree
import numpy as np
import random

def pareto_frontier_multi(myArray):
    # Sort on first dimension
    myArray = myArray[myArray[:,0].argsort()]
    # Add first row to pareto_frontier
    pareto_frontier = myArray[0:1,:]
    # Test next row against the last row in pareto_frontier
    for row in myArray[1:,:]:
        if sum([row[x] >= pareto_frontier[-1][x]
                for x in range(len(row))]) == len(row):
            # If it is better on all features add the row to pareto_frontier
            pareto_frontier = np.concatenate((pareto_frontier, [row]))
    return pareto_frontier


def compute_distances(clients):
    distances = []
    for from_client in clients:
        row = []
        for to_client in clients:
          row.append(distance.euclidean(from_client, to_client))
        distances.append(row)
    return np.array(distances)

def sub_routes_to_chromosome(routes):
  individual = []
  for item in routes:
    for i in item:
      individual.append(i)
  return individual

def subroute_distance(sequence, distances):
    dist = 0
    key = 1
    seq_dict = dict()
    seq_dict[0] = 0
    for s in sequence:
      seq_dict[key] = s
      key +=1
    sorted_route = dict(sorted(seq_dict.items(), key=lambda x:x[1], reverse=True))
    # print(seq_dict)
    # print(sorted_route)
    Keys = list(sorted_route.keys())
    Values = list(sorted_route.values())

    # print("Keys => ",Keys)
    # print("Values => ",Values)

    rt = [0]+sequence+[0]
    for item in range(len(rt)-1):
      # print(rt[item], rt[item+1], Values.index(rt[item]), Values.index(rt[item+1]))
      key1 = Keys[Values.index(rt[item])]
      key2 = Keys[Values.index(rt[item+1])]
      # print("Keys => ", key1, key2)
      dist += distances[(key1, key2)]
      # print(rt[item], rt[item+1], "Keys => ", key1, key2, dist)

    return dist

def route_distance(sequence, distances):
    dist = 0
    routes = []
    key = 1
    seq_dict = dict()
    seq_dict[0] = 0
    for s in sequence:
      seq_dict[key] = s
      key +=1
    sorted_route = dict(sorted(seq_dict.items(), key=lambda x:x[1], reverse=True))
    # print(seq_dict)
    # print(sorted_route)
    Keys = list(sorted_route.keys())
    Values = list(sorted_route.values())

    # print("Keys => ",Keys)
    # print("Values => ",Values)

    i1 = 0
    for i in range(len(Keys)-1):
      if np.abs(Values[i+1]-Values[i])>1:
        # print("Found one sub-route", i, Values[i1:i+1])
        r = Values[i1:i+1]
        routes.append(r)
        i1 = i+1

    # print(routes)

    for rt in routes:
      rt = [0]+rt+[0]
      for item in range(len(rt)-1):
        # print(rt[item], rt[item+1], Values.index(rt[item]), Values.index(rt[item+1]))
        key1 = Keys[Values.index(rt[item])]
        key2 = Keys[Values.index(rt[item+1])]
        dist += distances[key1, key2]
        # print(rt[item], rt[item+1], "Keys => ", key1, key2, dist)

    return dist

def sub_route_identifier(sequence):
    routes = []
    key = 1
    seq_dict = dict()
    for s in sequence:
      seq_dict[key] = s
      key +=1

    Keys = list(seq_dict.keys())
    Values = list(seq_dict.values())
    i1 = 0
    for i in range(len(Keys)-1):
      if np.abs(Values[i+1]-Values[i])>1:
        r = Values[i1:i+1]
        routes.append(r)
        i1 = i+1
    routes.append(Values[i1:])
    return routes

def get_swap_dict(d):
    return {v: k for k, v in d.items()}


def driver_renumeration(route, distances, service_time, B, m1, m2, Q_r):
  distance = route_distance(route, distances)

  total_service_time = 0
  for r in range(len(route)):
    total_service_time += service_time[r]
  time_duration = distance+total_service_time

  additional_time = 0
  for item in Q_r:
      additional_time += service_time[r]
  total_service_time += additional_time
  W = total_service_time

  M = 0
  if time_duration <= B:
    M = (W/B)*time_duration*m1
  else:
    M = W*m1+((W/B)*time_duration-W)*m2

  return M


import copy
def RSM(individual, distances_matrix, client_demands, client_time_window, capacity, service_times, B, m1, m2):
  distances = []
  n_subroutes = []
  M_values = []

  v_cap = capacity
  routes = sub_route_identifier(individual)
  route_keys = dict(enumerate(individual))
  # print(routes)
  # print(route_keys)
  n = len(client_demands)
  # n = 5
  additional_cost = []

  max_pr = max(individual)
  sorted_values = sorted(list(route_keys.values()), reverse=True)
  # print(sorted_values)
  for i in range(n):
    Q_r = []
    v_cap = capacity
    left_demands = copy.deepcopy(client_demands)
    left_demands.pop(0)
    # print("before assigning the demanads",left_demands)

    for key in left_demands:
      mean_value = left_demands[key]
      sigma = np.random.uniform(0,0.33)*mean_value
      left_demands[key] = np.random.normal(mean_value, sigma)
      # rand = random.random()
      # if rand>0.5:
      #   left_demands[key] = mean_value + np.random.normal(mean_value, sigma)
      # else:
      #   left_demands[key] = mean_value - np.random.normal(mean_value, sigma)
    # print("after assigning the demands", left_demands)

    left_demands_keys = list(left_demands.keys())
    j = 0
    while len(left_demands_keys):
      indx = sorted_values[j]
      client_index = list(route_keys.keys())[list(route_keys.values()).index(indx)]+1
      # print(f"index={indx}" , f"client index={client_index}", left_demands[client_index], v_cap)
      if v_cap >= left_demands[client_index]:
        v_cap-= left_demands[client_index]
        left_demands.pop(client_index)
        j += 1
      else:
        # print("Not", left_demands[client_index], v_cap)
        ## serve the client partially
        left_demands[client_index] -= v_cap
        ## return to depot to refill the capacity
        v_cap = capacity
        Q_r.append(client_index)

      left_demands_keys = list(left_demands.keys())
      # print("===============", left_demands)
    additional_cost.append(Q_r)

    for item in Q_r:
      added_subroute = [item, 0, item]
      # print(added_subroute)
      added_distance = 0
      for item in range(len(added_subroute)-1):
        added_distance += distances_matrix[added_subroute[item], added_subroute[item+1]]
      # print("Added distance", added_distance)

    dist = route_distance(individual, distances_matrix) + added_distance
    distances.append(dist)
    n_subroutes.append(len(routes))
    obj_M = driver_renumeration(individual, distances_matrix, service_times, B, m1, m2, Q_r)
    M_values.append(obj_M)

  # print(additional_cost)
  # print(distances)
  # print(M_values)
  return np.mean(distances), np.mean(M_values), np.mean(n_subroutes)
