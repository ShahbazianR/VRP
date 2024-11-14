import copy
from scipy.spatial import distance
import numpy as np
import random
import time


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
                    # print("d1 =>", d1, "d2 =>", d2, d1==d2)
            j += 1
        ## End While
        HR = dict(sorted(HR.items(), key=lambda item: item[1]))
        # print(HR)
        # print("End of While #1")

        j = 0
        assigned_clients = [] ## assigned clients => C"
        while j < len(customres):
            i = list(HR.keys())[j]
            # print("i =>", i)

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
        # print("End of While #2")
        # print("HD => ", HD)
        # print("C\" => ", assigned_clients)
        
        for item in assigned_clients:
            customres.remove(item)

        j = 0
        while j < len(depots):
            d = depots[j]
            td = len(HD[d])
            if td >= int(len(coordinates)/len(depot_coords)):
                depots.remove(d)
            j += 1
        # print("End of While #3")
    return HD

## Route Composition Function
def route_compose(route, depot, c_coords, d_coords, c_tw, c_service, c_demand, v_info):
    # print("\nRoute =>", route)
    # print("Depot =>", depot)
    customers = copy.deepcopy(route)
    depot_coord = d_coords[depot]
    routes = [depot]
    routes_distance = 0
    current_load = 0
    current_time = 0
    time_step = 0

    vehicle_max_load = v_info['max_load'][0]
    vehicle_max_time = v_info['max_T'][0]

    while len(customers):
        ci = customers[0]

        if current_load < vehicle_max_load and current_time < vehicle_max_time: 
            ## current vehicle load is lower than its maximum => allowed to go for customers
            if routes[-1] == depot:
                lst_coord = depot_coord
            else:
                lst_coord = c_coords[routes[-1]]

            if c_tw[ci][0] <= time_step <= c_tw[ci][1]: ## time window constraint is correctly met
                dist = distance.euclidean(lst_coord, c_coords[ci])
            elif time_step <= c_tw[ci][0]:
                time_step = c_tw[ci][0]
                dist = distance.euclidean(lst_coord, c_coords[ci])

            if (current_load+c_demand[ci]) <= vehicle_max_load and (current_time+dist+c_service[ci]) <= vehicle_max_time: 
                routes.append(ci)
                time_step += c_service[ci]
                routes_distance += dist
                current_load += c_demand[ci]
                current_time += dist + c_service[ci]
                # print(time_step, current_load, current_time, routes)
                del customers[0] ## Remove the first item from the list => assigned customer
            else:
                routes.append(depot)
                current_load = 0
                current_time = 0  
        else:
            routes.append(depot)
            current_load = 0
            current_time = 0            

        ## End while
    routes.append(depot)
    # print(routes)
    return routes

## Algorithm 2
def create_routes(HD_, C, D, lamda, C_TW, C_service, c_demand, vehicles):
    R = dict() ## Solution set of MVPRTW
    HD = []
    for key in HD_:
        R[key] = [key]
        for item in HD_[key]:
            HD.append((key, item))
    # print("\nHD =>", HD, "\n")

    current_loads = dict(zip(range(len(D)), [0]*len(D)))
    current_times = dict(zip(range(len(D)), [0]*len(D)))
    routes_distances = dict(zip(range(len(D)), [0]*len(D)))
    time_steps = dict(zip(range(len(D)), [0]*len(D)))


    vehicle_max_load = vehicles['max_load'][0] ## The vehicles are Homogenous
    vehicle_max_time = vehicles['max_T'][0]

    while len(HD):
        j = 0
        rand = random.random()
        if rand <= lamda:
            j = random.randint(0, len(HD)-1)
    
        i = HD[j][1]
        d = HD[j][0]
        # print(HD[j], C_TW[i], c_demand[i])

        if current_loads[d] < vehicle_max_load and current_times[d] < vehicle_max_time: 
            ## current vehicle load is lower than its maximum => allowed to go for customers
            if R[d][-1] == d:
                lst_coord = D[d]
            else:
                lst_coord = C[R[d][-1]]

            if C_TW[i][0] <= time_steps[d] <= C_TW[i][1]: ## time window constraint is correctly met
                dist = distance.euclidean(lst_coord, C[i])
            elif time_steps[d] <= C_TW[i][0]:
                time_steps[d] = C_TW[i][0]
                dist = distance.euclidean(lst_coord, C[i])

            if ((current_loads[d]+c_demand[i]) <= vehicle_max_load) and ((current_times[d]+dist+C_service[i]) <= vehicle_max_time): 
                R[d].append(i)
                time_steps[d] += C_service[i]
                routes_distances[d] += dist
                current_loads[d] += c_demand[i]
                current_times[d] += dist + C_service[i]
                # print(time_steps[d], current_loads[d], current_times[d], R[d])
                HD.remove(HD[j])
            else:
                R[d].append(d)
                current_loads[d] = 0
                current_times[d] = 0  
        else:
            R[d].append(d)
            current_loads[d] = 0
            current_times[d] = 0            
        

    ## End of While
    
    for key in R:
        R[key].append(key)
    # print(R)
    return R


## Function to determine the initial Solution for each depot
def initial_solution(coordinates, data_depot, lamda, customers_tw, customers_service, customers_demand, vehicles):
    HD = generating_clusters(coordinates, data_depot)
    R = create_routes(HD, coordinates, data_depot, lamda, customers_tw, customers_service, customers_demand, vehicles)

    for key in R:
        routes_list = route_decompose(R[key])
        routes = []
        for item in routes_list:
            routes.append(item[1:-1])
        R[key] = routes
    
    return R


## Evaluation Function
## A solution 𝑠 is evaluated by a function 𝐹 (𝑠) that represents a weighted 
## sum of three objective functions to be minimized
def evaluation_function(solutions, C, D, C_demands, C_service, C_TW, vehicles_info):
    solution = copy.deepcopy(solutions)

    for key in solution:
        if len(solution[key])>=1:
            if type(solution[key][0]) == int:
                 solution[key] = [solution[key]]

    total_distances = 0 ## The total distances in all routes and depots => represented by L in the equation (12)
    No_vehicles = dict()
    alpha_list = [] ## The list of distances between each pair of customers 
    phi_k = dict()

    # depot_pat_distances = dict()
    depot_route_demand = dict() ## demands of the visited customers for each route VC
    depot_route_time = dict() ## time of the routes taken by the vehicle for each route T_k
    depot_route_delay = dict()

    for key in solution: ## Calculating the total distance value
        No_vehicles[key] = len(solution[key])

        # depot_pat_distances[key] = []
        depot_route_demand[key] = []
        depot_route_time[key] = []
        depot_route_delay[key] = []
        
        total_time = 0
        for route in solution[key]:
            # route = [key] + route + [key]
            route.insert(0, key)
            route.append(key)

            route_dist = 0
            route_serive_time = 0
            route_delay = 0
            ind_i = 0

            while ind_i < len(route)-1:
                i = route[ind_i]
                j = route[ind_i+1]
                    
                if i in D.keys() and j in C.keys():
                    route_dist += distance.euclidean(D[i], C[j])
                elif i in C.keys() and j in C.keys():
                    route_dist += distance.euclidean(C[i], C[j])
                elif i in C.keys() and j in D.keys():
                    route_dist += distance.euclidean(C[i], D[j])
                ind_i += 1

                if i in C.keys():
                    route_serive_time += C_service[i]
                    if total_time > C_TW[i][1]:
                        route_delay += total_time-C_TW[i][1]
                    elif total_time < C_TW[i][0]:
                        total_time = C_TW[i][0]

                total_time += route_dist
                total_time += route_serive_time

            depot_route_time[key].append(route_dist+route_serive_time)
            depot_route_delay[key].append(route_delay)

        for route in solution[key]:
            route_demand = 0
            for c in route:
                if c in C.keys():
                    route_demand += C_demands[c] 
            depot_route_demand[key].append(route_demand)
        
        depot_dist = 0
        for route in solution[key]:
            index = 0
            while index < len(route)-1:
                i = route[index]
                j = route[index+1]
                if i in D.keys() and j in C.keys():
                    depot_dist += distance.euclidean(D[i], C[j])
                elif i in C.keys() and j in C.keys():
                    dist = distance.euclidean(C[i], C[j])
                    depot_dist += dist
                    alpha_list.append(dist)
                elif i in C.keys() and j in D.keys():
                    depot_dist += distance.euclidean(C[i], D[j])
                index += 1
            ## End While
        ## End For
        total_distances += depot_dist
    ## End For
    
    L = total_distances
    V = sum(No_vehicles.values())
    beta = 0.001
    alpha = np.average(alpha_list)
    omega_Q = float(alpha/np.average(list(C_demands.values())))
    omega_T = omega_TW = float(alpha/np.average(list(C_service.values())))

    VC = vehicles_info['max_load'][0]
    MT = vehicles_info['max_T'][0]

    for d in solution.keys():
        for i in range(len(solution[d])):
            part_1 = omega_Q*max(0, depot_route_demand[d][i]-VC)
            part_2 = omega_T*max(0, depot_route_time[d][i]-MT)
            part_3 = omega_TW*depot_route_delay[d][i]
            phi_k[(d,i)] = part_1 + part_2 + part_3

    phi = sum(phi_k.values())
    F_s = beta*L + alpha*V+phi
    return F_s


######################################
####### Local Search Operators #######
######################################

## Intra-route operators
## swapping two customers in the same route
def swap(route_):
    route = copy.deepcopy(route_)
    if len(route) > 1:
        items = random.sample(range(0, len(route)), 2)
        route[items[0]], route[items[1]] = route[items[1]], route[items[0]]

    return route

## A customer is removed and reinserted in another position in the same route
def reinsertion(route_):
    route = copy.deepcopy(route_)
    if len(route) > 1:
        item = random.sample(list(route), 1)[0]
        rand_ind = random.randint(0, len(route)-1)
        route.remove(item)
        route.insert(rand_ind, item)
    return route


## two consecutive customers are removed and reinserted in another position in the same route
def or_opt2(route_):
    route = copy.deepcopy(route_)
    if len(route) > 2:
        ind = random.randint(0, len(route)-2)
        items = [route[ind], route[ind+1]]
        route.remove(items[0])
        route.remove(items[1])
        rand_ind = random.randint(0, len(route)-1)
        route.insert(rand_ind, items[0])
        route.insert(rand_ind+1, items[1])
    return route


## two non-adjacent edges are deleted and two others are added to generate a new route
def two_opt(route_):
    route = copy.deepcopy(route_)

    if len(route) > 4: ## A B C D E => the minumum length for two non-adjacent edges should be 5
        ind_1, ind_2 = sorted(random.sample(list(range(len(route)-1)), k=2))
        
        max_iter = 500
        iteration = 0
        ## to avoid two adjacent or overlapping edges
        while ind_2-ind_1 in [1,2] and iteration <= max_iter: 
            # print("while", ind_1, ind_2)
            ind_1, ind_2 = sorted(random.sample(list(range(len(route)-1)), k=2))
            iteration += 1

        if ind_2-ind_1 in [1, 2]:
            return route

        edge_1 = route[ind_1: ind_1+2]
        edge_2 = route[ind_2: ind_2+2]

        route_1 = route[0: ind_1]
        route_1 += [edge_1[0], edge_2[0]]
        route_1 += reversed(route[ind_1+2:ind_2])
        route_1 += [edge_1[1], edge_2[1]]
        route_1 += route[ind_2+2:]
        route = route_1
    
    return route


## Identifying the possible combinations of three elements for three-opt algorithm
def possible_segments(n):
    segments = list(((i, j, k) for i in range(n) for j in range(i + 2, n-1) for k in range(j + 2, n - 1 + (i > 0))))
    return segments

## three edges are excluded and all possibilities of exchange between them are tested to generate new routes.
def three_opt(route_, i, j, k):
    route = copy.deepcopy(route_)
    A = route[i: j]
    B = route[j: k]
    C = route[k: ]
    D = route[0: i]
    if D:
        A = D + A
        D = []

    new_mutations = set()
    ## for 3-opt algorithm, there are 8 different ways to create new routes
    ## among which, only 4 3-opt moves: the ones with >=2 reversed patterns 
    ## the others can be achieved by 2-opt moves
    ## case_1 = original route
    case_1 = route
    new_mutations.add(tuple(case_1))
    ## case_2 = A'BC
    case_2 = list(reversed(A)) + B + C
    new_mutations.add(tuple(case_2))
    # ABC'
    case_3 = A + B + list(reversed(C))
    new_mutations.add(tuple(case_3))
    # A'BC'
    case_4 = list(reversed(A)) + B + list(reversed(C))
    new_mutations.add(tuple(case_4))
    # A'B'C
    case_5 = list(reversed(A)) + list(reversed(B)) + C
    new_mutations.add(tuple(case_5))
    # AB'C
    case_6 = A + list(reversed(B)) + C
    new_mutations.add(tuple(case_6))
    # AB'C'
    case_7 = A + list(reversed(B)) + list(reversed(C))
    new_mutations.add(tuple(case_7))
    # A'B'C'
    case_8 = list(reversed(A)) + list(reversed(B)) + list(reversed(C))
    new_mutations.add(tuple(case_8))

    mutations = []
    for item in new_mutations:
        item = list(item)
        mutations.append(item)

    return mutations

# sample_route = list(range(10))
# segments = possible_segments(len(sample_route))
# (i, j, k) = segments[-10]
# print(three_opt(sample_route, i, j, k))


## Inter-route operators
## This function is also applicable as an inter depot operator 
def swap_1_1(route_1, route_2, v_i, v_j):
    route_i = copy.deepcopy(route_1)
    route_j = copy.deepcopy(route_2)
    vi_ind = route_i.index(v_i)
    vj_ind = route_j.index(v_j)

    route_i[vi_ind] = v_j
    route_j[vj_ind] = v_i
    return route_i, route_j


# sample_route1 = list(range(0, 10))
# sample_route2 = list(range(20, 30))
# print(sample_route1, 2)
# print(swap_1_1(sample_route1, sample_route2, 2, 25)) 


## This function is also applicable as an inter depot operator 
def shift_1_0(route_1, route_2, vi):
    route_i  = copy.deepcopy(route_1)
    route_j = copy.deepcopy(route_2)
    route_i.remove(vi)

    rand_ind = random.randint(0, len(route_j))
    route_j.insert(rand_ind, vi)

    return route_i, route_j

# sample_route1 = list(range(0, 10))
# sample_route2 = list(range(20, 30))
# print(sample_route1, sample_route2)
# print(shift_1_0(sample_route1, sample_route2, 2)) 

## Inter-depot operators
## Inter-route operators + the following functions
def swap_2_2_interdepot(route_1, route_2, vi, vj):
    route_i = copy.deepcopy(route_1)
    route_j = copy.deepcopy(route_2)

    vi_ind = route_i.index(vi)
    vj_ind = route_j.index(vj)
    seg_1 = route_i[vi_ind: vi_ind+2]
    seg_2 = route_j[vj_ind: vj_ind+2]

    route_i.remove(seg_1[0]) 
    route_i.remove(seg_1[1]) ## To Fix list index out of range

    route_j.remove(seg_2[0])
    route_j.remove(seg_2[1])

    route_i.insert(vi_ind, seg_2[0])
    route_i.insert(vi_ind+1, seg_2[1])
    route_j.insert(vj_ind, seg_1[0])
    route_j.insert(vj_ind+1, seg_1[1])

    return route_i, route_j


# sample_route1 = list(range(0, 10))
# sample_route2 = list(range(20, 30))
# print(sample_route1, sample_route2)
# print(swap_2_2_interdepot(sample_route1, sample_route2, 2, 25)) 


## Destruction and Reinstruction
# def dest_reinst(solutions_list):
#     print(solutions_list)
#     if len(solutions_list) <=1:
#         return solutions_list

#     path_lens = dict()
#     for sol in solutions_list:
#         path_lens[tuple(sol)] = len(sol)
    
#     sh_path_len = min(path_lens.values())
#     path_lens_reversed = {v: k for k,v in path_lens.items()}
#     sh_path = list(path_lens_reversed[sh_path_len])

#     solutions_list.remove(sh_path)

#     while sh_path:
#         c = sh_path.pop()
#         rand_route_ind = random.randint(0, len(solutions_list)-1)
#         rand_ind = random.randint(0, len(solutions_list[rand_route_ind])-1)
#         solutions_list[rand_route_ind].insert(rand_ind, c)
        
#     return solutions_list


def dest_reinst(solutions_list):
    solution = copy.deepcopy(solutions_list)

    if len(list(solution.values())[0]) <=1:
        return solution

    sh_custs = None
    sh_depot = None
    for sol in solution.keys():
        for route in solution[sol]:
            if not sh_custs:
                sh_custs = route
                sh_depot = sol
            elif len(route) < len(sh_custs):
                sh_custs = route
                sh_depot = sol
    
    solution[sh_depot].remove(sh_custs)
    
    while sh_custs:
        rand_key = random.sample(list(solution.keys()), k=1)[0]
        if len(solution[rand_key]):
            rand_route = random.randint(0, len(solution[rand_key])-1)
            if len(solution[rand_key][rand_route])>1:
                c = sh_custs.pop()
                rand_ind = random.sample(list(range(0, len(solution[rand_key][rand_route]))), k=1)[0]
                ## print(c, rand_key, rand_route)
                solution[rand_key][rand_route].insert(rand_ind, c)
            else:
                c = sh_custs.pop()
                ## print(c, rand_key, rand_route, len(solution[rand_key][rand_route]))
                solution[rand_key][rand_route].append(c)
    return solution


# sample_route1 = list(range(0, 10))
# sample_route2 = list(range(20, 30))
# print(sample_route1, sample_route2)
# print(dest_reinst({0:[sample_route1, sample_route2, [0, 1, 2]]})) 

#######################################
########### Main Algorithms ###########
#######################################

### LSO = {N1, ... , N11} : All Local Search Operators
### PO = {N8(shift_1_0_interdepot), N9(swap_1_1_interdepot), N11(eliminate_smallest_route)} : shake operators

def shake(solution:dict, v:int, p:int, C, D, C_TW, C_service, C_demands, vehicles_info):
    if v not in [8, 9, 11]:
        return solution
    
    s = copy.deepcopy(solution)
    i = 1
    while i<=p:
        if v == 8:
            depots = random.sample(list(s.keys()), k=2)
            if len(s[depots[0]]) and len(s[depots[1]]):
                route_0_index = random.sample(list(range(0, len(s[depots[0]]))), k=1)[0]
                route_0 = s[depots[0]][route_0_index]
                route_1_index = random.sample(list(range(0, len(s[depots[1]]))), k=1)[0]
                route_1 = s[depots[1]][route_1_index]

                vi = random.sample(route_0, k=1)[0]
                vj = random.sample(route_1, k=1)[0]

                si, sj = swap_1_1(route_0, route_1, vi, vj)

                if si and sj:
                    if max(len(si), len(sj)) <=2:
                        s[depots[0]].remove(route_0)
                        s[depots[0]].append(si)
                        
                        s[depots[1]].remove(route_1)
                        s[depots[1]].append(sj)
                    else:
                        si_ = route_check(depots[0], si, C, D, C_TW, C_service, C_demands, vehicles_info)
                        sj_ = route_check(depots[1], sj, C, D, C_TW, C_service, C_demands, vehicles_info)
                        # print("Route Check >>", si_, sj_)

                        s[depots[0]].remove(route_0)
                        for ind in range(len(si_)):
                            s[depots[0]].append(si_[ind])

                        s[depots[1]].remove(route_1)
                        for ind in range(len(sj_)):
                            s[depots[1]].append(sj_[ind])

        elif v==9:
            depots = sorted(random.sample(list(s.keys()), k=2))
            if len(s[depots[0]]) and len(s[depots[1]]):
                route_0_index = random.sample(list(range(0, len(s[depots[0]]))), k=1)[0]
                route_0 = s[depots[0]][route_0_index]
                route_1_index = random.sample(list(range(0, len(s[depots[1]]))), k=1)[0]
                route_1 = s[depots[1]][route_1_index]

                vi = random.sample(route_0, k=1)[0]

                si, sj = shift_1_0(route_0, route_1, vi)

                if si and sj:
                    if max(len(si), len(sj)) <=2:
                        s[depots[0]].remove(route_0)
                        s[depots[0]].append(si)
                        
                        s[depots[1]].remove(route_1)
                        s[depots[1]].append(sj)
                    else:
                        si_ = route_check(depots[0], si, C, D, C_TW, C_service, C_demands, vehicles_info)
                        sj_ = route_check(depots[1], sj, C, D, C_TW, C_service, C_demands, vehicles_info)

                        s[depots[0]].remove(s[depots[0]][route_0_index])
                        for ind in range(len(si_)):
                            s[depots[0]].append(si_[ind])

                        s[depots[1]].remove(s[depots[1]][route_1_index])
                        for ind in range(len(sj_)):
                            s[depots[1]].append(sj_[ind])

        elif v==11:
            s = dest_reinst(s)
            for key in s:
                depot_routes = []
                for route in s[key]:
                    split_route = route_check(key, route, C, D, C_TW, C_service, C_demands, vehicles_info)
                    for item in split_route:
                        depot_routes.append(item)
                s[key] = depot_routes

        i += 1

    return s


## Proposed Local Search
## ALS: Algorithm 4 
def ALS(s, s_, success, itLS, maxLS1, C, D, C_demands, C_service, C_TW, vehicles_info):
    
    if itLS < maxLS1:
        s_2, success = RVND(s, s_, success, C, D, C_demands, C_service, C_TW, vehicles_info) 
    else:
        s_2, success = single_local_search(s, s_, success, C, D, C_demands, C_service, C_TW, vehicles_info)

    return s_2, success

## Algorithm 5: Update Success
def update_success(s, s_, s_2, v, success, C, D, C_demands, C_service, C_TW, vehicles_info):
    # print("Update Success ", v, success)
    if v not in success.keys():
        success[v] = 0
    eval_s2 = evaluation_function(s_2, C, D, C_demands, C_service, C_TW, vehicles_info) 
    eval_s = evaluation_function(s, C, D, C_demands, C_service, C_TW, vehicles_info)
    eval_s_ = evaluation_function(s_, C, D, C_demands, C_service, C_TW, vehicles_info)
    if eval_s2 < eval_s:
        success[v] += 15
    elif eval_s2 < eval_s_:
        success[v] += 5
    return success


def path_evaluation(route, C, D, C_demands, C_service, C_TW, vehicles_info):
    total_distances = 0 ## The total distances in all routes and depots => represented by L in the equation (12)
    alpha_list = [] ## The list of distances between each pair of customers 
    phi_k = dict()

    No_vehicles = 1

    route_demand = 0
    route_delay = 0
    
    total_time = 0
    route_dist = 0
    route_serive_time = 0
    route_delay = 0
    ind_i = 0
    while ind_i < len(route)-1:
        i = route[ind_i]
        j = route[ind_i+1]
            
        if i in D.keys() and j in C.keys():
            route_dist += distance.euclidean(D[i], C[j])
        elif i in C.keys() and j in C.keys():
            route_dist += distance.euclidean(C[i], C[j])
        elif i in C.keys() and j in D.keys():
            route_dist += distance.euclidean(C[i], D[j])
        ind_i += 1

        if i in C.keys():
            route_serive_time += C_service[i]
            if total_time > C_TW[i][1]:
                route_delay += total_time-C_TW[i][1]
            elif total_time < C_TW[i][0]:
                total_time = C_TW[i][0]

        total_time += route_dist
        total_time += route_serive_time

    depot_route_time = route_dist+route_serive_time
    depot_route_delay = route_delay
    
    route_demand = 0
    for c in route:
        if c in C.keys():
            route_demand += C_demands[c] 
    depot_route_demand = route_demand

    depot_dist = 0
    index = 0
    while index < len(route)-1:
        i = route[index]
        j = route[index+1]
        if i in D.keys() and j in C.keys():
            depot_dist += distance.euclidean(D[i], C[j])
        elif i in C.keys() and j in C.keys():
            dist = distance.euclidean(C[i], C[j])
            depot_dist += dist
            alpha_list.append(dist)
        elif i in C.keys() and j in D.keys():
            depot_dist += distance.euclidean(C[i], D[j])
        index += 1
        ## End While
    total_distances += depot_dist

    L = total_distances
    V = No_vehicles
    beta = 0.001
    alpha = np.average(alpha_list)
    omega_Q = float(alpha/np.average(list(C_demands.values())))
    omega_T = omega_TW = float(alpha/np.average(list(C_service.values())))

    VC = vehicles_info['max_T'][0]
    MT = vehicles_info['max_load'][0]

    part_1 = omega_Q*max(0, depot_route_demand-VC)
    part_2 = omega_T*max(0,depot_route_time-MT)
    part_3 = omega_TW*depot_route_delay
    phi_k = part_1 + part_2 + part_3

    phi = phi_k
    F_s_path = beta*L + alpha*V+phi
    return F_s_path


def route_check(depot, route_, C, D, C_TW, C_service, C_demands, vehicles):
    # route = [depot]+route_+[depot]
    # print("Route check Function:", route_)
    route = copy.deepcopy(route_)
    if len(route) <=2:
        return [route]

    VC = vehicles['max_T'][0]
    MT = vehicles['max_load'][0]
    valid = True

    route_demand = 0
    route_delay = 0
    total_time = 0
    route_dist = 0
    route_serive_time = 0
    route_delay = 0
    ind_i = 0
    while ind_i < len(route)-1:
        i = route[ind_i]
        j = route[ind_i+1]
            
        if i in D.keys() and j in C.keys():
            route_dist += distance.euclidean(D[i], C[j])
        elif i in C.keys() and j in C.keys():
            route_dist += distance.euclidean(C[i], C[j])
        elif i in C.keys() and j in D.keys():
            route_dist += distance.euclidean(C[i], D[j])
        ind_i += 1

        if i in C.keys():
            route_serive_time += C_service[i]
            if total_time > C_TW[i][1]:
                route_delay += total_time-C_TW[i][1]
            elif total_time < C_TW[i][0]:
                total_time = C_TW[i][0]

        total_time += route_dist
        total_time += route_serive_time
    
    route_demand = 0
    for c in route:
        if c in C.keys():
            route_demand += C_demands[c] 

    if route_demand > VC or total_time > MT:
        valid = False

    refactored_routes = []
    if not valid:
        route_time = 0 
        demand = 0
        ind_i = 0
        ind_j = 0
        while ind_i < len(route)-1:
            i = route[ind_i]
            j = route[ind_i+1]

            if i in C.keys():
                demand += C_demands[i] 
                route_serive_time += C_service[i]
                if route_time < C_TW[i][0]:
                    route_time = C_TW[i][0]
            route_time += route_dist
            route_time += route_serive_time

            if demand > VC or route_time > MT:
                if ind_i > ind_j:
                    refactored_routes.append(route[ind_j: ind_i+1])
                    ind_j = ind_i+1
                    demand = 0
                    route_time = 0  
            ind_i += 1

        refactored_routes.append(route[ind_j: ])
        return refactored_routes
    else:
        return [route]


## Algorithm 6: VND
## s: Solution; Assumingly, s is a dictionary of original routes associated with depots
def RVND(s, s_, success, C, D, C_demands, C_service, C_TW, vehicles_info):

    ##print(">> RVND ")
    N_ = random.randint(1, N)
    v = 1

    s_2 = copy.deepcopy(s_)
    while v <= N_:
        s_2 = copy.deepcopy(s_)
        ## print("===================================")
        ## print(f">> RVND Iteration V = {v}")
        for key in s_2: 
            s_2[key] = [i for i in s_2[key] if i is not None]
            s_2[key] = [i for i in s_2[key] if i!=[]]

        # print(f"Current Solution of the Iteration:\n {s_2}")

        if v == 1:
            depot = random.sample(list(s_.keys()), k=1)[0]
            if len(s_[depot]) >= 1:
                route_index = random.sample(list(range(0, len(s_[depot]))), k=1)[0]
                route = s_[depot][route_index]

                if route in s_2[depot]:
                    s_2[depot].remove(route)
                route_ = swap(route)
                s_2[depot].insert(route_index, route_)

        elif v == 2:
            depot = random.sample(list(s_.keys()), k=1)[0]
            if len(s_[depot]):
                route_index = random.sample(list(range(0, len(s_[depot]))), k=1)[0]
                route = s_[depot][route_index]
                if route in s_2[depot]:
                    s_2[depot].remove(route)
                s_2[depot].insert(route_index, reinsertion(route))

        elif v == 3:
            depot = random.sample(list(s_.keys()), k=1)[0]    
            if len(s_[depot]):
                route_index = random.sample(list(range(0, len(s_[depot]))), k=1)[0]
                route = s_[depot][route_index]
                if route in s_2[depot]:
                    s_2[depot].remove(route)
                s_2[depot].insert(route_index, or_opt2(route))

        elif v == 4:
            depot = random.sample(list(s_.keys()), k=1)[0]
            if len(s_[depot]) >= 1:
                route_index = random.sample(list(range(0, len(s_[depot]))), k=1)[0]
                route = s_[depot][route_index]
                if route in s_2[depot]:
                    s_2[depot].remove(route)
                s_2[depot].insert(route_index, two_opt(route))

        elif v == 5:
            depot = random.sample(list(s_.keys()), k=1)[0]         
            if len(s_[depot]) > 1:
                route_index = random.sample(list(range(0, len(s_[depot]))), k=1)[0]
                if len(s_2[depot]) > route_index:
                    route = s_2[depot][route_index]
                else:
                    route = s_2[depot][0]
                
                segments = possible_segments(len(route))
                three_opt_results = []
                for item in segments:
                    (i, j, k) = item
                    three_opt_results.append(three_opt(route, i, j, k))

                three_opt_costs_min = np.inf
                three_opt_result_min = []
                for result in three_opt_results:
                    for path in result:
                        cost = path_evaluation(path, C, D, C_demands, C_service, C_TW, vehicles_info)
                        if cost <= three_opt_costs_min:
                            three_opt_costs_min = cost
                            three_opt_result_min = path

                if three_opt_result_min:                   
                    s_2[depot].remove(route)
                    s_2[depot].append(three_opt_result_min)

        elif v == 6: ## same depot
            depot = random.sample(list(s_.keys()), k=1)[0]       
            if len(s_[depot]) >= 2:
                routes_indices = random.sample(list(range(0, len(s_[depot]))), k=2)
                routes = [s_[depot][routes_indices[0]], s_[depot][routes_indices[1]]]
                if len(routes[0]) and len(routes[1]):
                    vi = random.sample(routes[0], k=1)[0]
                    vj = random.sample(routes[1], k=1)[0]
                    si, sj = swap_1_1(routes[0], routes[1], vi, vj)
                    s_2[depot][routes_indices[0]] = si
                    s_2[depot][routes_indices[1]] = sj

        elif v == 7: ## same depot
            depot = random.sample(list(s_.keys()), k=1)[0]
            if len(s_[depot]) >= 2:
                routes_indices = random.sample(list(range(0, len(s_[depot]))), k=2)
                routes = [s_[depot][routes_indices[0]], s_[depot][routes_indices[1]]]

                vi = random.sample(routes[0], k=1)[0]
                si, sj = shift_1_0(routes[0], routes[1], vi)
                s_2[depot][routes_indices[0]] = si
                s_2[depot][routes_indices[1]] = sj

        elif v == 8: ## different depots
            depots = random.sample(list(s_.keys()), k=2)
            if len(s_[depots[0]]) and len(s_[depots[1]]):
                route_0_index = random.sample(list(range(0, len(s_[depots[0]]))), k=1)[0]
                route_0 = s_[depots[0]][route_0_index]
                route_1_index = random.sample(list(range(0, len(s_[depots[1]]))), k=1)[0]
                route_1 = s_[depots[1]][route_1_index]
                # print("Routes =>", route_0, route_1)
                if route_0 and route_1:
                    vi = random.sample(route_0, k=1)[0]
                    vj = random.sample(route_1, k=1)[0]

                    si, sj = swap_1_1(route_0, route_1, vi, vj)
                    # print("Swap 1 1 I:", vi, si,"J:", vj, sj)
                    if si and sj:
                        if max(len(si), len(sj)) <=2:
                            s_2[depots[0]].remove(route_0)
                            s_2[depots[0]].append(si)
                            
                            s_2[depots[1]].remove(route_1)
                            s_2[depots[1]].append(sj)
                        else:
                            
                            si_ = route_check(depots[0], si, C, D, C_TW, C_service, C_demands, vehicles_info)
                            sj_ = route_check(depots[1], sj, C, D, C_TW, C_service, C_demands, vehicles_info)
                            ## print("Route Check >>", si_, sj_)

                            s_2[depots[0]].remove(route_0)
                            if type(si_[0]) == list:
                                for ind in si_:
                                    s_2[depots[0]].append(ind)
                            else:
                                s_2[depots[0]].append(si_)

                            s_2[depots[1]].remove(route_1)
                            if type(sj_[0]) == list:
                                for ind in sj_:
                                    s_2[depots[1]].append(ind)
                            else:
                                s_2[depots[1]].append(sj_)

        elif v==9:
            depots = sorted(random.sample(list(s_.keys()), k=2))
            if len(s_[depots[0]]) and len(s_[depots[1]]):
                route_0_index = random.sample(list(range(0, len(s_[depots[0]]))), k=1)[0]
                route_0 = s_[depots[0]][route_0_index]
                route_1_index = random.sample(list(range(0, len(s_[depots[1]]))), k=1)[0]
                route_1 = s_[depots[1]][route_1_index]
                # print("Routes =>", route_0, route_1)
                vi = random.sample(route_0, k=1)[0]

                si, sj = shift_1_0(route_0, route_1, vi)

                if si and sj:
                    if max(len(si), len(sj)) <=2:
                        s_2[depots[0]].remove(route_0)
                        s_2[depots[0]].append(si)
                        
                        s_2[depots[1]].remove(route_1)
                        s_2[depots[1]].append(sj)
                    else:
                        si_ = route_check(depots[0], si, C, D, C_TW, C_service, C_demands, vehicles_info)
                        sj_ = route_check(depots[1], sj, C, D, C_TW, C_service, C_demands, vehicles_info)
                        # print("Route Check >>", si_, sj_)

                        s_2[depots[0]].remove(s_2[depots[0]][route_0_index])
                        if type(si_[0]) == list:
                            for ind in si_:
                                s_2[depots[0]].append(ind)
                        else:
                            s_2[depots[0]].append(si_)

                        s_2[depots[1]].remove(s_2[depots[1]][route_1_index])
                        if type(sj_[0]) == list:
                            for ind in sj_:
                                s_2[depots[1]].append(ind)
                        else:
                            s_2[depots[1]].append(sj_)

                    
                        # print(">>> ", s_2[depots[0]])
                        # print(">>> ", s_2[depots[1]])

        elif v == 10:
            depots = random.sample(list(s_.keys()), k=2)
            if len(s_[depots[0]]) and len(s_[depots[1]]):
                route_0_index = random.sample(list(range(0, len(s_[depots[0]]))), k=1)[0]
                route_0 = s_[depots[0]][route_0_index]
                route_1_index = random.sample(list(range(0, len(s_[depots[1]]))), k=1)[0]
                route_1 = s_[depots[1]][route_1_index]

                if min(len(route_0), len(route_1)) > 2:
                    vi = random.sample(route_0[0:len(route_0)-1], k=1)[0]
                    vj = random.sample(route_1[0:len(route_1)-1], k=1)[0]

                    si, sj = swap_2_2_interdepot(route_0, route_1, vi, vj)

                    s_2[depots[0]][route_0_index] = si
                    s_2[depots[1]][route_1_index] = sj
                elif len(route_0) == len(route_1) == 2:
                    vi = route_0[0]
                    vj = route_1[0]

                    si, sj = swap_2_2_interdepot(route_0, route_1, vi, vj)

                    s_2[depots[0]][route_0_index] = si
                    s_2[depots[1]][route_1_index] = sj

        elif v == 11:
            s_2 = dest_reinst(s_)
            for key in s_2:
                depot_routes = []
                for route in s_2[key]:
                    split_route = route_check(key, route, C, D, C_TW, C_service, C_demands, vehicles_info)
                    if type(split_route[0]) == list:
                        for item in split_route:
                            depot_routes.append(item)
                    elif type(split_route[0]) == int:
                        depot_routes.append(split_route)

                s_2[key] = depot_routes
        ################################################
        ################################################
        for key in s_2:
            if len(s_2[key])>=1:
                if type(s_2[key][0]) == int:
                    s_2[key] = [s_2[key]]

        eval_s2 = evaluation_function(s_2, C, D, C_demands, C_service, C_TW, vehicles_info)
        eval_s_ = evaluation_function(s_, C, D, C_demands, C_service, C_TW, vehicles_info)

        success = update_success(s, s_, s_2, v, success, C, D, C_demands, C_service, C_TW, vehicles_info)

        if eval_s2 < eval_s_:
            s_ = copy.deepcopy(s_2)
            v = 1
        else:
            v += 1
    return s_, success


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

## Single Local Search Algorithm
def single_local_search(s, s_, success, C, D, C_demands, C_service, C_TW, vehicles_info):
    v = roullette(success)
    s_2 = copy.deepcopy(s_)

    ## print(">> Single Local Search")

    if v == 1:
        depot = random.sample(list(s_.keys()), k=1)[0]
        if len(s_[depot]) >= 1:
            route_index = random.sample(list(range(0, len(s_[depot]))), k=1)[0]
            route = s_[depot][route_index]
            s_2[depot].remove(s_[depot][route_index])
            s_2[depot].insert(route_index, swap(route))

    elif v == 2:
        depot = random.sample(list(s_.keys()), k=1)[0]
        if len(s_[depot]):
            route_index = random.sample(list(range(0, len(s_[depot]))), k=1)[0]
            route = s_[depot][route_index]
            s_2[depot].remove(route)
            s_2[depot].insert(route_index, reinsertion(route))

    elif v == 3:
        depot = random.sample(list(s_.keys()), k=1)[0]         
        if len(s_[depot]):
            route_index = random.sample(list(range(0, len(s_[depot]))), k=1)[0]
            route = s_[depot][route_index]
            s_2[depot].remove(route)
            s_2[depot].insert(route_index, or_opt2(route))

    elif v == 4:
        depot = random.sample(list(s_.keys()), k=1)[0]
        if len(s_[depot]) >= 1:
            route_index = random.sample(list(range(0, len(s_[depot]))), k=1)[0]
            route = s_[depot][route_index]
            s_2[depot].remove(route)
            s_2[depot].insert(route_index, two_opt(route))

    elif v == 5:
        depot = random.sample(list(s_.keys()), k=1)[0]         
        if len(s_[depot]) >= 1:
            route_index = random.sample(list(range(0, len(s_[depot]))), k=1)[0]
            route = s_2[depot][route_index]
            
            segments = possible_segments(len(route))
            three_opt_results = []
            for item in segments:
                (i, j, k) = item
                three_opt_results.append(three_opt(route, i, j, k))

            three_opt_costs_min = np.inf
            three_opt_result_min = []
            for result in three_opt_results:
                for path in result:
                    cost = path_evaluation(path, C, D, C_demands, C_service, C_TW, vehicles_info)
                    if cost <= three_opt_costs_min:
                        three_opt_costs_min = cost
                        three_opt_result_min = path

            if three_opt_result_min:                           
                s_2[depot].remove(route)
                s_2[depot].append(three_opt_result_min)

    elif v == 6: ## same depot
        depot = random.sample(list(s_.keys()), k=1)[0]       
        if len(s_[depot]) >= 2:
            routes_indices = random.sample(list(range(0, len(s_[depot]))), k=2)
            routes = [s_[depot][routes_indices[0]], s_[depot][routes_indices[1]]]
            if len(routes[0]) and len(routes[1]):
                vi = random.sample(routes[0], k=1)[0]
                vj = random.sample(routes[1], k=1)[0]
                si, sj = swap_1_1(routes[0], routes[1], vi, vj)
                s_2[depot][routes_indices[0]] = si
                s_2[depot][routes_indices[1]] = sj

    elif v == 7: ## same depot
        depot = random.sample(list(s_.keys()), k=1)[0]
        if len(s_[depot]) >= 2:
            routes_indices = random.sample(list(range(0, len(s_[depot]))), k=2)
            routes = [s_[depot][routes_indices[0]], s_[depot][routes_indices[1]]]

            if len(routes[0]):
                vi = random.sample(routes[0], k=1)[0]
                si, sj = shift_1_0(routes[0], routes[1], vi)
                s_2[depot][routes_indices[0]] = si
                s_2[depot][routes_indices[1]] = sj

    elif v == 8: ## different depots
        depots = random.sample(list(s_.keys()), k=2)
        if len(s_[depots[0]]) and len(s_[depots[1]]):
            route_0_index = random.sample(list(range(0, len(s_[depots[0]]))), k=1)[0]
            route_0 = s_[depots[0]][route_0_index]
            route_1_index = random.sample(list(range(0, len(s_[depots[1]]))), k=1)[0]
            route_1 = s_[depots[1]][route_1_index]

            if len(route_0) and len(route_1):
                vi = random.sample(route_0, k=1)[0]
                vj = random.sample(route_1, k=1)[0]

                si, sj = swap_1_1(route_0, route_1, vi, vj)

                if si and sj:
                    if max(len(si), len(sj)) <=2:
                        s_2[depots[0]].remove(route_0)
                        s_2[depots[0]].append(si)
                        
                        s_2[depots[1]].remove(route_1)
                        s_2[depots[1]].append(sj)
                    else:
                        si_ = route_check(depots[0], si, C, D, C_TW, C_service, C_demands, vehicles_info)
                        sj_ = route_check(depots[1], sj, C, D, C_TW, C_service, C_demands, vehicles_info)

                        s_2[depots[0]].remove(route_0)
                        if type(si_[0]) == list:
                            for ind in si_:
                                s_2[depots[0]].append(ind)
                        else:
                            s_2[depots[0]].append(si_)

                        s_2[depots[1]].remove(route_1)
                        if type(sj_[0]) == list:
                            for ind in sj_:
                                s_2[depots[1]].append(ind)
                        else:
                            s_2[depots[1]].append(sj_)

    elif v==9:
        depots = sorted(random.sample(list(s_.keys()), k=2))
        if len(s_[depots[0]]) and len(s_[depots[1]]):
            route_0_index = random.sample(list(range(0, len(s_[depots[0]]))), k=1)[0]
            route_0 = s_[depots[0]][route_0_index]
            route_1_index = random.sample(list(range(0, len(s_[depots[1]]))), k=1)[0]
            route_1 = s_[depots[1]][route_1_index]
            if len(route_0) and len(route_1):
                vi = random.sample(route_0, k=1)[0]

                si, sj = shift_1_0(route_0, route_1, vi)
                if si and sj:
                    if max(len(si), len(sj)) <=2:
                        s_2[depots[0]].remove(route_0)
                        s_2[depots[0]].append(si)
                        
                        s_2[depots[1]].remove(route_1)
                        s_2[depots[1]].append(sj)
                    else:
                        si_ = route_check(depots[0], si, C, D, C_TW, C_service, C_demands, vehicles_info)
                        sj_ = route_check(depots[1], sj, C, D, C_TW, C_service, C_demands, vehicles_info)

                        s_2[depots[0]].remove(s_2[depots[0]][route_0_index])
                        if type(si_[0]) == list:
                            for ind in si_:
                                s_2[depots[0]].append(ind)
                        else:
                            s_2[depots[0]].append(si_)

                        s_2[depots[1]].remove(s_2[depots[1]][route_1_index])
                        if type(sj_[0]) == list:
                            for ind in sj_:
                                s_2[depots[1]].append(ind)
                        else:
                            s_2[depots[1]].append(sj_)


    elif v == 10:
        depots = random.sample(list(s_.keys()), k=2)
        if len(s_[depots[0]]) and len(s_[depots[1]]):
            route_0_index = random.sample(list(range(0, len(s_[depots[0]]))), k=1)[0]
            route_0 = s_[depots[0]][route_0_index]
            route_1_index = random.sample(list(range(0, len(s_[depots[1]]))), k=1)[0]
            route_1 = s_[depots[1]][route_1_index]

            if min(len(route_0), len(route_1)) > 2:
                vi = random.sample(route_0[0:len(route_0)-1], k=1)[0]
                vj = random.sample(route_1[0:len(route_1)-1], k=1)[0]

                si, sj = swap_2_2_interdepot(route_0, route_1, vi, vj)

                s_2[depots[0]][route_0_index] = si
                s_2[depots[1]][route_1_index] = sj
                
            elif len(route_0) == len(route_1) == 2:
                vi = route_0[0]
                vj = route_1[0]

                si, sj = swap_2_2_interdepot(route_0, route_1, vi, vj)
                s_2[depots[0]][route_0_index] = si
                s_2[depots[1]][route_1_index] = sj

    elif v == 11:
        s_2 = dest_reinst(s_)
        for key in s_2:
            depot_routes = []
            for route in s_2[key]:
                split_route = route_check(key, route, C, D, C_TW, C_service, C_demands, vehicles_info)
                for item in split_route:
                    depot_routes.append(item)
            s_2[key] = depot_routes

    success = update_success(s, s_, s_2, v, success, C, D, C_demands, C_service, C_TW, vehicles_info)
    for key in s_2:
        if len(s_2[key])>=1:
            if type(s_2[key][0]) == int:
                s_2[key] = [s_2[key]]
            
    return s_2, success


## SGVNSALS Algorithm => The method prospoed in the paper
### LSO = {N1, ... , N11} : All Local Search Operators
### PO = {N8(shift_1_0_interdepot), N9(swap_1_1_interdepot), N11(eliminate_smallest_route)} : shake operators
### N = {N1, ..., N11}: All the Neighborhood Operators
def SGVNSALS(C, D, iterMax, maxTime, maxLevel, pLS1, pLS2, als:bool, TW_C, ServiceT_C, Demand_C, Vehicle_info, lamda):
    """
    C: Customers 
    D: Depots
    als: Defines the Algorithm (Either SGVNSALS[True] or SGVNS[False])
    """

    print("\n======== Initial Solution ========")
    s = initial_solution(C, D, lamda, TW_C, ServiceT_C, Demand_C, Vehicle_info)
    print("Initial Solution => ", s)
    print("Evaluation Initial Solution =>", evaluation_function(s, C, D, Demand_C, ServiceT_C, TW_C, Vehicle_info))

    success = dict() ## Initialize Success for Neighborhood Operators
    for i in range(1,N):
        success[i] = 0

    p = 1 ## Initial Shake Level
    v = 1 ## Index For The First Neighborhood
    iter = 0 ## Iteration counter without improvement
    itLS = 0 ## Counter for the local search
    maxLS1 = pLS1*iterMax ## Number of the RVND iterations    
    maxLS2 = (pLS1 + pLS2) * iterMax; ## Number of the SingleLocalSearch iterations

    start_op_time = time.time()
    current_op_time = time.time()
    maxoperationtime = 1*60*60

    print("\n======== Iterations ========")
    while iter < iterMax and (current_op_time-start_op_time) < maxoperationtime:
        if int(current_op_time-start_op_time) % 120 == 0:
            print(f"Iteration {iter} and OP: {(current_op_time-start_op_time)}")
        current_op_time = time.time()
        
        s_ = shake(s, v, p, C, D, TW_C, ServiceT_C, Demand_C, Vehicle_info)

        if als: ## als == True: SGVNSALS Algorithm
            s_2, success = ALS(s, s_, success, itLS, maxLS1, C, D, Demand_C, ServiceT_C, TW_C, Vehicle_info)
        else:   ## als == False: SGVNS Algorithm
            s_2, success = RVND(s, s_, success, C, D, Demand_C, ServiceT_C, TW_C, Vehicle_info)

        eval_s_2 = evaluation_function(s_2, C, D, Demand_C, ServiceT_C, TW_C, Vehicle_info)
        eval_s = evaluation_function(s, C, D, Demand_C, ServiceT_C, TW_C, Vehicle_info)
        if eval_s_2 < eval_s:
            s = s_2
            v = 1 ## Return to the first neighborhood
            p = 1 ## Return to the first shake level
            iter = 0 ## Reset the iteration counter without improvement
        else:
            p += 1; ## Increase the shake level
            iter += 1 ## Increase the iteration counter without improvement  
        ## End If
        
        if p > maxLevel:
            v += 1 ## Move to the next neighborhood
            p = 1 ## Return to the first shake level
        ## End If

        if v > 3:  ## 3 == |PO|
            v = 1 ## Return to the first neighborhood
            p = 1 ## Return to the first shake level
        ## End If
            
        itLS += 1
        if itLS > maxLS2:
            itLS = 0
        ## End If
    ## End While
    return s
