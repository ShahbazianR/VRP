import numpy as np
from abc import ABC, abstractmethod
from functions import *


class Individual(ABC):
    def __init__(self, value=None, init_params=None):
        if value is not None:
            self.value = value
        else:
            self.value = self._random_init(init_params)

    @abstractmethod
    def pair(self, other, pair_params):
        pass

    @abstractmethod
    def mutate(self, mutate_params):
        pass

    @abstractmethod
    def _random_init(self, init_params):
        pass


class Optimization(Individual):
    def pair(self, other, pair_params):
        return Optimization(pair_params['alpha'] * self.value + (1 - pair_params['alpha']) * other.value)

    def mutate(self, mutate_params):
        self.value += np.random.normal(0, mutate_params['rate'], mutate_params['dim'])
        for i in range(len(self.value)):
            if self.value[i] < mutate_params['lower_bound']:
                self.value[i] = mutate_params['lower_bound']
            elif self.value[i] > mutate_params['upper_bound']:
                self.value[i] = mutate_params['upper_bound']

    def _random_init(self, init_params):
        return np.random.uniform(init_params['lower_bound'], init_params['upper_bound'], init_params['dim'])

##
##class Population:
##    def __init__(self, size, fitness, individual_class, init_params):
##        self.fitness = fitness
##        self.individuals = [individual_class(init_params=init_params) for _ in range(size)]
##        self.individuals.sort(key=lambda x: self.fitness(x))
##
##    def replace(self, new_individuals):
##        size = len(self.individuals)
##        self.individuals.extend(new_individuals)
##        self.individuals.sort(key=lambda x: self.fitness(x))
##        self.individuals = self.individuals[-size:]
##
##    def get_parents(self, n_offsprings):
##        mothers = self.individuals[-2 * n_offsprings::2]
##        fathers = self.individuals[-2 * n_offsprings + 1::2]
##
##        return mothers, fathers


class Evolution:
    def __init__(self, pool_size, fitness, individual_class, n_offsprings, pair_params, mutate_params, init_params):
        self.pair_params = pair_params
        self.mutate_params = mutate_params
        self.pool = population(pool_size, fitness, individual_class, init_params)
        self.n_offsprings = n_offsprings

    def step(self):
        mothers, fathers = self.pool.get_parents(self.n_offsprings)
        offsprings = []

        for mother, father in zip(mothers, fathers):
            offspring = mother.pair(father, self.pair_params)
            offspring.mutate(self.mutate_params)
            offsprings.append(offspring)

        self.pool.replace(offsprings)


import random
class chromosome_class():
  def __init__(self, length, max_priority, initialize=False):
    self.length = length
    self.max_priority = max_priority
    self.chromosome = []

    if initialize:
      self.initialize()

  def initialize(self):
    max_prior = self.max_priority
    for i in range(1,self.length+1):
      rand_ = random.random()
      if rand_ <0.4:
        max_prior -= 1
        self.chromosome.append(max_prior)
      else:
        self.chromosome.append(max_prior)
      max_prior -= 1

    routes = sub_route_identifier(self.chromosome)
    random.shuffle(routes)
    self.chromosome = sub_routes_to_chromosome(routes)

from numpy.random import randint
class population():
  def __init__(self, generation, length, vehicle_range, demands, vehicle_capacity, time_windows, distances, B, initialize=False):
    self.size = generation
    self.length = length
    self.vehicle_range = vehicle_range
    self.client_demands = demands
    self.vehicle_capacity = vehicle_capacity
    self.client_time_windows = time_windows
    self.client_number = len(self.client_demands)
    self.N_r = self.client_number
    self.distances = distances
    self.B = B
    
    #self.vehicle_number = randint(1,self.vehicle_range)
    self.vehicle_number = randint(3,self.vehicle_range)
    print("vehicle number => ",self.vehicle_number)
    self.max_priority = self.client_number + self.N_r-1
    print("max priority => ", self.max_priority)

    self.population = []

    if initialize:
      self.initialize()

  def demand_check(self, sequence):
    total_demand = 0
    for i in range(len(sequence)):
      total_demand += self.client_demands[i]
    return total_demand <= self.vehicle_capacity

  def initialize(self):
    print("initializing the population ... \n")
    max_nv = self.vehicle_number
    i = 0
    rep = 0
    while i <= self.size and rep <= 1000:
      rep = rep+1
      ch = chromosome_class(self.length, self.max_priority, initialize=True)
      routes = sub_route_identifier(ch.chromosome)
      demand_flag = True
      for subroute in routes:
        if not self.demand_check(subroute):
          demand_flag = False
          break
      time_flag = True
      for subroute in routes: ## we assume the time is equal to the distance
        dist = subroute_distance(subroute, self.distances)
        if dist >= self.B:
          time_flag = False
          break

      if demand_flag and time_flag and len(routes)<= max_nv: ## the chromosome is accepted
        if i == 0:
          max_nv = len(routes)
        self.population.append(ch.chromosome)
        i += 1


  def create(self, tree, n_generation):
    pop_H = []
    self.n_generation = n_generation
    # print("H generated population => ",int(self.n_generation/2))
    for i in range(int(self.n_generation/2)):
      ch = chromosome_class(self.length, self.max_priority, initialize=True)
      prd = tree.predict([ch.chromosome])[0]
      # while prd==0 or ch.chromosome in pop_H:
      rep = 0
      while prd==0 and rep<10:
        rep += 1
        ch = chromosome_class(self.length, self.max_priority, initialize=True)
        prd = tree.predict([ch.chromosome])[0]
      pop_H.append(ch.chromosome)
      # print(i, prd, ch.chromosome)

    pop_Random = []
    for i in range(int(self.n_generation/2), self.n_generation):
      ch = chromosome_class(self.length, self.max_priority, initialize=True)
      pop_Random.append(ch.chromosome)
    return pop_H+pop_Random

  def partial_swapping(self, individual):
    routes = sub_route_identifier(individual)
    # print("Routes before the partial swap => ", routes)
    if len(routes)>=2:
      index1 = random.randint(0, len(routes)-1) ## Choose two subroutes to partial swap their randomly selected segments
      index2 = random.randint(0, len(routes)-1)
      while index1 == index2:
        index2 = random.randint(0, len(routes)-1)

      min_index = min(index1, index2)
      max_index = max(index1, index2)
      sub_routes = [routes[min_index], routes[max_index]]
      # print("Sub routes =>", sub_routes)

      segments= []
      for route in sub_routes:
        if len(route)>=2:
          idx1 = random.randint(0, len(route)-1)
          idx2 = random.randint(0, len(route)-1)
          while idx1 == idx2:
            idx1 = random.randint(0, len(routes)-1)
          min_index = min(idx1, idx2)
          max_index = max(idx1, idx2)
          segments.append([min_index, max_index, route[min_index: max_index]])
        else:
          segments.append([0, 1, route])

      # print("Segments => ", segments)
      value_0 = segments[0][2][:]
      value_1 = segments[1][2][:]
      # print("values => ", value_0, value_1)
      sub_routes[0][segments[0][0]:segments[0][1]] = value_1
      sub_routes[1][segments[1][0]:segments[1][1]] = value_0
      # print("Swapped routes => ", sub_routes[0], sub_routes[1])

      routes[index1] = sub_routes[0]
      routes[index2] = sub_routes[1]
      # print("Routes after partial swap = >", routes)
    else:
        return individual
    return sub_routes_to_chromosome(routes)

  def priority_fix(self, routes):
    # print("Inside Priority Fix => ", routes)

    individual_dict = dict()
    individual = []
    for subroute in routes:
      individual += subroute
    individual_dict = dict(enumerate(individual))
    index_dict = dict(enumerate(sorted(list(individual_dict.values()), reverse=True)))
    index_dict = get_swap_dict(index_dict)
    # print("individual_dict => ", individual_dict)
    # print("index_dict => ", index_dict)

    paths = []
    for route in routes:
      path = []
      for i in route:
        path.append(index_dict[i])
      paths.append(path)
    # print(paths)

    max_pr = len(individual_dict.keys())+len(routes)-1
    # print(len(individual_dict.keys()), len(routes), max_pr)
    for path in paths:
      d = dict(sorted(dict(enumerate(path)).items(), key=lambda x:x[1])).values()
      d = dict(enumerate(d))
      for key in d:
        path[path.index(d[key])] = max_pr
        max_pr -= 1
      max_pr -= 1
    # print("Fixed Routes => ", paths)
    return paths

  def merging_sub_routes(self, individual, distances):
    routes = sub_route_identifier(individual)
    # print("inside merging subroutes", routes)
    if len(routes)>=2:
      subroute_dists = dict()
      for route_index in range(len(routes)):
        subroute_dists[route_index] = subroute_distance(routes[route_index], distances)
      # print(subroute_dists)

      sorted_dists = sorted(subroute_dists.items(), key=lambda x:x[1])
      # print(sorted_dists)

      min_index = min(sorted_dists[0][0], sorted_dists[1][0])
      max_index = max(sorted_dists[0][0], sorted_dists[1][0])

      merged_route = routes[min_index]+routes[max_index]
      # print("merged route => ", merged_route)
      routes[min_index] = merged_route
      routes.remove(routes[max_index])
      routes = self.priority_fix(routes)

    return sub_routes_to_chromosome(routes)

  def split_routes(self, individual, distances):
    routes = sub_route_identifier(individual)
    # print("inside splitting subroutes", routes)

    subroute_dists = dict()
    for route_index in range(len(routes)):
      subroute_dists[route_index] = subroute_distance(routes[route_index], distances)
    # print(subroute_dists)

    sorted_dists = sorted(subroute_dists.items(), key=lambda x:x[1], reverse=True)
    # print(sorted_dists)
    route_index = sorted_dists[0][0]
    longest_route = routes[route_index]
    # print("Longest sub-route =>", longest_route)

    rand_position = random.randint(1,len(longest_route)-1)
    split_1 = longest_route[0:rand_position]
    split_2 = longest_route[rand_position:]
    routes.insert(route_index, split_1)
    routes.insert(route_index+1, split_2)
    routes.remove(longest_route)
    routes = self.priority_fix(routes)
    # print("Rand position and split sections => ", rand_position, split_1, split_2)
    # print("routes after splitting the longest route => ", routes)
    return sub_routes_to_chromosome(routes)

  def random_suffle(self, individual):
    routes = sub_route_identifier(individual)
    # print("inside random shuffle", routes)

    rand_indx = random.randint(0, len(routes)-1)
    random.shuffle(routes[rand_indx])
    # print("After shuffling => ", rand_indx, routes)
    return sub_routes_to_chromosome(routes)

  def heuristic_operations(self, individual, distances, swap_rate, merge_rate, shuffle_rate):
    swap_rand = random.random()
    merge_rand = random.random()
    shuffle_rand = random.random()

    resulted_individual = individual

    if swap_rand < swap_rate:
      resulted_individual = self.partial_swapping(resulted_individual)
      # print("Individual after partial swapping => ", resulted_individual)
      # print("============= Partial Swapping Done ================")
    elif merge_rand < merge_rate:
      resulted_individual = self.merging_sub_routes(resulted_individual, distances)
      # print("Individual after merging => ", resulted_individual)
      # print("============== Merging Done ===============")
    else:
      resulted_individual = self.split_routes(resulted_individual, distances)
      # print("Individual after Splitting => ", resulted_individual)
      # print("============== Splitting the longest Done ====================")

    if shuffle_rand < shuffle_rate:
      resulted_individual = self.random_suffle(resulted_individual)
      # print("Individual after Shuffling => ", resulted_individual)
      # print("============== Random Shuffling Done ====================")

    # print(resulted_individual)
    return resulted_individual
  
