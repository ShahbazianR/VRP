import copy
from scipy.spatial import distance
import numpy as np
import random


N = 12

## Function to decompose a solution route for a depot
def route_decompose(route):
    depot = route[0]
    routes = []
    depot_indices = [i for i, item in enumerate(route) if item==depot]
    for i in range(len(depot_indices)-1):
        routes.append(route[depot_indices[i]: depot_indices[i+1]+1])
    return routes

## Algorithm 1
def generating_clusters(coordinates, depot_coords):
    customres = copy.deepcopy(list(coordinates.keys())) ## customers => C'
    depots = copy.deepcopy(list(depot_coords.keys())) ## depots => D'
    
    HD = dict(zip(range(len(depots)), [list()]*len(depots))) ## set if (i,d) of customers assigned to depot d

    while len(customres): ## customers => C'
        A = dict(zip(range(len(depots)), [list()]*len(depots))) ## pairs of (i,d) where d is the closest depot to the customer i 
        B = dict(zip(range(len(depots)), [list()]*len(depots))) ## pairs of (i,d) where d is the second closest depot to the customer i    
        HR = dict(zip(range(len(customres)), [np.inf]*len(customres))) ## set of pairs (i, 𝑟𝑑1𝑑2) where 𝑟𝑑1𝑑2 is the ratio of proximity of customer 𝑖 to the two closest depots

        j = 0
        while j < len(customres):
            i = customres[j]
            i_coords = coordinates[i]

            id_dists = dict()
            for d in range(len(depots)):
                d_coords = depot_coords[d]
                dist =  distance.euclidean(d_coords, i_coords)
                id_dists[d] = dist
            
            dist_reversed = {v: k for k, v in id_dists.items()}
            if len(id_dists):
                c_d1i = min(id_dists.values())
                d1 = dist_reversed[c_d1i]
                A[d1] = list(set(A[d1] + [i]))
                dist_copy = copy.deepcopy(id_dists)

                if len(dist_copy)>1: ## This Block is added: Code Modified to avoid a forever interminable loop
                    dist_copy.pop(d1) ## causes problem 

                dist_reversed = {v: k for k, v in dist_copy.items()}
                if len(dist_copy):
                    c_d2i = min(dist_copy.values())
                    d2 = dist_reversed[c_d2i]
                    B[d2] = list(set(B[d2] + [i]))
                    rd1d2 = float(c_d1i/c_d2i)
                    HR[i] = rd1d2

            j += 1
        ## End While
        HR = dict(sorted(HR.items(), key=lambda item: item[1]))

        j = 0
        assigned_clients = [] ## assigned clients => C"
        while j < len(customres):
            i = list(HR.keys())[j]

            for di in A.keys():
                if i in A[di]:
                    d1 = di
                elif i in B[di]:
                    d2 = di

            td1 = len(HD[d1])
            td2 = len(HD[d2])
            if td1 < int(len(coordinates)/len(depot_coords)):
                HD[d1] = HD[d1] + [i]
                assigned_clients.append(i)
            elif td2 < int(len(coordinates)/len(depot_coords)):
                HD[d2] = HD[d2] + [i]
                assigned_clients.append(i)
            else: ## This Block is added: Code Modified to avoid a forever interminable loop
                HD[d1] = HD[d1] + [i]
                assigned_clients.append(i)
            j += 1
        ## End While
            
        for item in assigned_clients:
            customres.remove(item)

        j = 0
        while j < len(depots):
            d = depots[j]
            td = len(HD[d])
            if td >= int(len(coordinates)/len(depot_coords)):
                depots.remove(d)
            j += 1
    return HD
