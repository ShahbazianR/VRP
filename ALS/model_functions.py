import random, copy
import numpy as np
import math
from utils import *

def generate_initial_solution(C, D, demands, TWs, STs, max_capacity, max_time, distance_matrix):
    # depot = random.sample(list(D.keys()), k=1)[0]
    depot = list(D.keys())[0]

    solution = list()
    solution.append(depot)

    total_nodes = list(D.keys()) + list(C.keys())
    unvisited_c = list(C.keys())
    Q = max_capacity
    T = max_time

    min_dist = (0, np.inf)
    for c in list(C.keys()):
        distance = distance_matrix[(depot, c)]
        if distance < min_dist[1]:
            min_dist = (c, distance)

    solution.append(min_dist[0])
    total_nodes.remove(min_dist[0])
    unvisited_c.remove(min_dist[0])

    solution_demand = 0
    solution_time = 0
    last_depot = depot
    while len(unvisited_c):
        p_nodes = dict()
        for node in unvisited_c:
            dist = distance_matrix[(node, solution[-1])]
            if (solution_demand + demands[node] <= Q) and (solution_time + STs[node] + dist <= T):
                p_nodes[node] = dist
        p_nodes = sorted(p_nodes.items(), key=lambda x:x[1])

        if len(p_nodes)==0:
            solution.append(last_depot)
            p_nodes = dict()
            depots = list(D.keys())
            depots.remove(last_depot)
            for node in depots:
                p_nodes[node] = distance_matrix[(node, solution[-1])]
            p_nodes = sorted(p_nodes.items(), key=lambda x:x[1])
            node = p_nodes[0][0]
            solution.append(node)
            solution_demand = 0
            solution_time = 0
            Q = max_capacity
            T = max_time
            last_depot = node

        else:
            node = p_nodes[0][0]
            solution.append(node)
            total_nodes.remove(node)
            unvisited_c.remove(node)
            solution_demand += demands[node]
            solution_time += STs[node] + p_nodes[0][1]
            Q -= demands[node]
            T -= (STs[node] + p_nodes[0][1])
        
    if solution[-1] not in D.keys():
        solution.append(last_depot)
    
    return solution


def routes_decompose(solution, n_depots):
    depots = list(range(n_depots))
    routes = []
    route = [solution[0]]
    for node_ind in range(1, len(solution)):
        node = solution[node_ind]
        if node in depots:
            if len(route):
                route.append(node)
                routes.append(route)
                route = []
            else:
                route.append(node)
        else:
            route.append(node)

    return routes

################################################
############ Destruction Operators #############
################################################

def random_removal(n_custs, n_depots, k=1):
    return random.sample(list(range(n_depots+1, n_custs+n_depots)), k)


def random_route_removal(solution, n_depots, n):
    depots = list(range(n_depots))
    routes = routes_decompose(solution, n_depots)

    solution_ = copy.deepcopy(solution)
    request_list = []
    i = 0
    while i <= n:
        route_ind = random.sample(list(range(len(routes))), k=1)[0]
        route = routes[route_ind]
        for item in route:
            if item not in depots:
                request_list.append(item)
                if item in solution_:
                    solution_.remove(item)
        i += len(route)

    return request_list, solution_


def shaw_removal(solution, D, H, distance_matrix, TW, Demands): ##  Ropke & Pisinger (2006)
    alpha_1 = 0.4
    alpha_2 = 0.8
    alpha_3 = 0.3

    solution_ = copy.deepcopy(solution)

    i = random.sample(solution, k=1)[0]
    while i in D.keys():
        i = random.sample(solution, k=1)[0]
    
    request_list = []
    request_list.append(i)

    d_max = max(distance_matrix.values())
    q_max = max(Demands.values())

    j_ = None
    min_val = np.inf
    for j in solution:
        if j not in D.keys():
            term1 = alpha_1 * (distance_matrix[(j,i)]/d_max)
            term2 = alpha_2 * (np.abs(TW[i][0]-TW[j][0])/H)
            term3 = alpha_3 * (np.abs(Demands[i]-Demands[j])/q_max)
            total_val = term1 + term2 + term3

            if total_val < min_val:
                j_ = j
                min_val = total_val

    solution_.remove(j_)
    return j_, solution_


def worst_removal(solution, D, distance_matrix):
    solution_ = copy.deepcopy(solution)
    i_ = None
    max_val = -np.inf
    for i in solution[1:-1]:
        if i not in D.keys():
            j = solution[solution.index(i)-1]
            k = solution[solution.index(i)+1]
            d_ji = distance_matrix[(j,i)]
            d_ik = distance_matrix[(i,k)]
            d_jk = distance_matrix[(j,k)]
            total_val = d_ji + d_ik - d_jk

            if total_val > max_val:
                i_ = i
                max_val = total_val
    solution_.remove(i_)
    return i_, solution_


## Proposed Removal Operators
def cost_reducing_removal(solution, D, distance_matrix, TW, ST, max_iter=1):
    solution_ = copy.deepcopy(solution)
    routes = routes_decompose(solution, len(D.keys()))
    
    request_list = []

    iter = 0
    while iter <= max_iter:
        iter += 1

        rand_route_ind = random.randint(0, len(routes)-1)
        rand_route = routes[rand_route_ind]

        for v in rand_route[1:-1]:
            i1 = rand_route[rand_route.index(v)-1] ## previous node
            j1 = rand_route[rand_route.index(v)+1] ## next node

            for route in routes:
                for i2 in route:
                    j2 = route[route.index(i2)+1] 
                    ## if vertex v can be inserted in arc (i, j)
                    tw_compatibility = timewinodw_compatibility_insertion(route, v, route.index(i2), distance_matrix, TW, ST)
                    if tw_compatibility:
                        if (distance_matrix[(i1,v)] + distance_matrix[(v,j1)] + distance_matrix[(i2,j2)]) > (distance_matrix[(i1,j1)] + distance_matrix[(i2,v)] + distance_matrix[(v,j2)]):
                            if v in solution_:
                                request_list.append(v)
                                solution_.remove(v)
        
    return request_list, solution_


def exchange_reducing_removal(solution, D, distance_matrix, TW, ST, max_iter=1):
    request_list = set()
    solution_ = copy.deepcopy(solution)
    
    routes = routes_decompose(solution, len(D.keys()))

    arcs = [(i,j) for i in solution for j in solution if i not in D.keys() and j not in D.keys() and i!=j]

    iter = 0
    while iter <= max_iter:
        iter += 1

        rand_nodes = random.sample(arcs, k=1)[0]
        v1, v2 = rand_nodes
        v1_route = v2_route = []

        for route in routes:
            if v1 in route:
                v1_route = route
            if v2 in route:
                v2_route = route

        i1 = v1_route[v1_route.index(v1)-1] ## previous node of v1
        i2 = v2_route[v2_route.index(v2)-1] ## previous node of v2

        j1 = v1_route[v1_route.index(v1)+1] ## next node of v1
        j2 = v2_route[v2_route.index(v2)+1] ## next node of v2
        
        if (distance_matrix[(i1,v1)] + distance_matrix[(v1,j1)] + distance_matrix[(i2,v2)] + distance_matrix[(v2,j2)]) > (distance_matrix[(i1,v2)] + distance_matrix[(v2,j1)] + distance_matrix[(i2,v1)] + distance_matrix[(v1,j2)]):
            v1_tw_compatibility = timewinodw_compatibility_insertion(v1_route, v1, v1_route.index(i1), distance_matrix, TW, ST)
            v2_tw_compatibility = timewinodw_compatibility_insertion(v2_route, v2, v2_route.index(i2), distance_matrix, TW, ST)

            if v1_tw_compatibility:
                request_list.add(v1)
                if v1 in solution_:
                    solution_.remove(v1)

            if v2_tw_compatibility:
                request_list.add(v2)
                if v2 in solution_:
                    solution_.remove(v2)

    return list(request_list), solution_
                

################################################
############ Insertion Operators ###############
################################################

## Greedy insertion rule: 
## p(j, i, k) = argmin{dji + dik − djk}, j, k ∈ Vd ∪ V_0^pi ∈ L
def greedy_insertion(solution, D, request_list, distance_matrix):
    routes = routes_decompose(solution, len(D.keys()))

    p_values = dict()
    p_results = dict()
    for i in request_list:
        p_values[i] = (None, None, None, np.inf)
        for route in routes:
            if len(route) == 2:
                j = route[0]
                k = route[1]
                d_values = distance_matrix[(k,i)] + distance_matrix[(i,k)] - distance_matrix[(j,k)]
                if d_values < p_values[i][3]:
                    p_values[i] = (routes.index(route), j, k, d_values)
                    p_results[i] = (routes.index(route), j, k, d_values)
            else:
                for j in route[1:-1]:
                    k = route[route.index(j)+1]
                    d_values = distance_matrix[(k,i)] + distance_matrix[(i,k)] - distance_matrix[(j,k)]
                    if d_values < p_values[i][3]:
                        p_values[i] = (routes.index(route), j, k, d_values)
                        p_results[i] = (routes.index(route), j, k, d_values)

    for key in p_results.keys():
        routes[p_results[key][0]].insert(routes[p_results[key][0]].index(p_results[key][1])+1, key)
        request_list.remove(key)

    solution_ = []
    for route in routes:
            solution_.extend(route)

    return request_list, solution_


## https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3587400/
## Ropke and Pisinger: An Adaptive Large Neighborhood Search Heuristic
def regret_3_insertion(solution, D, request_list, distance_matrix): 
    routes = routes_decompose(solution, len(D.keys()))
    L = copy.deepcopy(request_list)

    while len(L):
        cost_values = dict()
        routes_min_positions = dict()
        ci = dict()
        for i in L:
            cost_values[i] = dict()
            routes_min_positions[i] = dict()
            for route in routes:
                route_index = routes.index(route)
                routes_min_positions[i][route_index] = (None, None, np.inf)
                for j in route[:-1]:
                    k = route[route.index(j)+1]
                    d_values = distance_matrix[(k,i)] + distance_matrix[(i,k)] - distance_matrix[(j,k)]
                    if d_values < routes_min_positions[i][routes.index(route)][2]:
                        routes_min_positions[i][routes.index(route)] = (j, route.index(j)+1, d_values)
                
                route_inserted = copy.deepcopy(route)
                route_inserted.insert(route.index(j), i)
                total_dist = tour_distance(route_inserted, distance_matrix)
                cost_values[i][route_index] = total_dist

            cost_values[i] = sorted(cost_values[i].items(), key=lambda x:x[1])
            ci[i] = np.abs(cost_values[i][2][1]-cost_values[i][0][1]) + np.abs(cost_values[i][1][1]-cost_values[i][0][1])

        ci = sorted(ci.items(), key=lambda x:x[1], reverse=True)

        request = ci[0][0]
        route_to_insert = cost_values[request][0][0]
        min_position = routes_min_positions[request][route_to_insert][1]
        routes[route_to_insert].insert(min_position, request)

        L.remove(request)

    solution_ = []
    for route in routes:
            solution_.extend(route)

    return L, solution_


#################################################################
############## Adaptive and Compatibility Methods ###############
#################################################################

destruction_pool = [random_removal, random_route_removal, shaw_removal, worst_removal]
insertion_pool = [greedy_insertion, regret_3_insertion]
def adaptive_method(probs, op, cost_S=1, cost_S_=1):
    if op in destruction_pool:
        sum_probs = 0
        for item in probs:
            if item in destruction_pool:
                sum_probs += probs[item]
        probs[op] *= probs[op]/sum_probs
    
    elif op in insertion_pool:
        probs[op] += (cost_S - cost_S_)/cost_S
    
    return probs


def depot_variation(solution, D, distance_matrix): 
    ## Condition: di0,i1 + diu− 1,iu > dj,i1 + diu− 1,j
    routes = routes_decompose(solution, len(D.keys()))

    for route in routes:
        i0 = route[0]
        i1 = route[1]
        iu_1 = route[-2]
        iu = route[-1]
        for depot in D.keys():
            if (distance_matrix[(i0, i1)] + distance_matrix[(iu_1, iu)]) > (distance_matrix[(depot, i1)] + distance_matrix[(iu_1,depot)]):
                route[0] = route[-1] = depot

    solution_ = []
    for route in routes:
        # if len(route)>2:
            solution_.extend(route)
    return solution_


def timewinodw_compatibility_insertion(route, c, position, distance_matrix, TW, ST): 
    ## max{eμ(L) + tiμc, ac} ≤ min{lμ+1(L) − tciμ+1, bc}
    # print(c, position, route)

    route_inserted = copy.deepcopy(route)
    route_inserted.insert(position, c)

    mu = route[position]
    mu_1 = route[position+1]
    
    e_u = TW[mu][0]
    l_u = TW[mu][1]

    t_muc = distance_matrix[(mu,c)] + ST[c]
    t_cmu_1 = distance_matrix[(c, mu_1)]

    a_c = TW[c][0]
    b_c = TW[c][1]

    return max(e_u+t_muc, a_c) <= min(l_u-t_cmu_1, b_c)


################################################
################ Main Algorithm ################
################################################

## Roullette Method 
def roullette(op_probs):
    probs = dict()
    fitness_sum = sum([op_probs[key] for key in op_probs])
    previous_probability = 0
    for op in op_probs:
        fitness_op = op_probs[op]
        probs[op] = previous_probability + (fitness_op/fitness_sum)
        previous_probability = probs[op]

    random_number = random.random()
    selected_op = list(op_probs.keys())[0] ## Default value is the first one
    for key in op_probs.keys():
        if probs[key] > random_number:
            break; 
        selected_op = key
    return selected_op


def ALS(C, D, Demands, TTW, ST, MAX_C, MAX_T, distance_matrix, T, beta, N, remvoal_gamma):
    intitial_solution = generate_initial_solution(C, D, Demands, TTW, ST, MAX_C, MAX_T, distance_matrix)
    print("Initial Solution > ", routes_decompose(intitial_solution, len(D.keys())))

    H = MAX_T
    cost_reducing_dest_max_it = 100

    current_solution = copy.deepcopy(intitial_solution) ## S'
    optimal_solution = copy.deepcopy(intitial_solution) ## S*
    S = copy.deepcopy(intitial_solution)

    ## initializing the probabilities for the operators
    dest_op_probs = dict(zip(destruction_pool, [0.1]*len(destruction_pool)))
    ins_op_probs = dict(zip(insertion_pool, [0.1]*len(insertion_pool)))

    iterations = N
    theta_iter = T
    request_list = []
    while iterations > 0 :
        if iterations % 500 == 0:
            print(iterations)
        iterations -= 1
        theta_iter = beta*theta_iter
        ## Destruction Component
        dest_operator = roullette(dest_op_probs)
        # print("Destruction Operator:", dest_operator)
        if dest_operator == random_removal:
            r = dest_operator(len(C), len(D), k=remvoal_gamma)
            request_list.extend(r)
            for item in r:
                if item in S:
                    S.remove(item)

        elif dest_operator == random_route_removal:
            r, S = dest_operator(S, len(D), n=remvoal_gamma)
            request_list.extend(r)
        
        elif dest_operator == shaw_removal:
            j, S = dest_operator(S, D, H, distance_matrix, TTW, Demands)
            request_list.append(j)

        elif dest_operator == worst_removal:
            i, S = dest_operator(S, D, distance_matrix)
            request_list.append(i)
        
        elif dest_operator == cost_reducing_removal:
            r, S = dest_operator(S, D, distance_matrix, TTW, ST, max_iter=cost_reducing_dest_max_it)
            request_list.extend(r)

        elif dest_operator == exchange_reducing_removal:
            r, S = dest_operator(S, D, distance_matrix, TTW, ST, max_iter=cost_reducing_dest_max_it)
            request_list.extend(r)
        # print(f"\nDestruction {dest_operator} Phase Finished.\nRequest List:{request_list}, Solution:{routes_decompose(S, len(D.keys()))}")
        
        request_list = list(set(request_list))

        ## Insertion Operator
        insert_operator = roullette(ins_op_probs)
        # print("Insertion Operator:", insert_operator)
        if insert_operator == greedy_insertion:
            request_list, S = insert_operator(S, D, request_list, distance_matrix)
        
        elif insert_operator == regret_3_insertion:
            request_list, S = insert_operator(S, D, request_list, distance_matrix)
        # print(f"\nInsertion {insert_operator} Phase Finished.\nRequest_list:{request_list}, Solution:{routes_decompose(S, len(D.keys()))}")

        # print("\nDepot Variation Applied.")
        S = depot_variation(S, D, distance_matrix)
        # print("Solution Routes:", routes_decompose(S, len(D.keys())))
        
        S = route_breaking(S, C, D, MAX_T, distance_matrix, TTW, ST) ### TODO: Added to ensure the Max route time

        ## Update Component
        total_cost_S = tour_distance(S, distance_matrix) 
        total_cost_S_ = tour_distance(current_solution, distance_matrix) ## current solution

        # print("S_:", total_cost_S_, "S:", total_cost_S)
        if not len(request_list):
            if total_cost_S < total_cost_S_:
                current_solution = copy.deepcopy(S)
                # print("Current Solution Replaced.")
                # print(routes_decompose(current_solution, len(D.keys())))

                ## adapting the probabilities
                adaptive_method(dest_op_probs, dest_operator)
                adaptive_method(ins_op_probs, insert_operator, total_cost_S, total_cost_S_)

                total_cost_S_ = tour_distance(current_solution, distance_matrix) ## current solution
                total_cost_S_star = tour_distance(optimal_solution, distance_matrix)
                
                # print("S*:", total_cost_S_star, "S_:", total_cost_S_)
                if total_cost_S_ < total_cost_S_star:
                    optimal_solution = copy.deepcopy(current_solution)
                    # print("Optimal Solution Replaced:\n", routes_decompose(optimal_solution, len(D.keys())))
            else:
                z = random.random()
                if z < math.e**(-(total_cost_S-total_cost_S_)/theta_iter):
                    current_solution = S

        # print("=======================")
        # print("Optimal Solution:",routes_decompose(optimal_solution, len(D.keys())))
        # print("=======================")

    # print("Optimal Solution:",routes_decompose(optimal_solution, len(D.keys())))
    return optimal_solution, request_list


def route_breaking(solution, C, D, Max_T, distance_matrix, TTW, ST):
    routes = routes_decompose(solution, len(D))

    for route in routes:
        if len(route) > 2:
            route_time = route_time_eval(route, distance_matrix, TTW, ST)
            if route_time > Max_T:
                break_point = 2
                depot = route[0]
                route_time = route_time_eval(route[0:break_point] + [depot], distance_matrix, TTW, ST)
                while route_time < Max_T or break_point == len(route):
                    break_point += 1
                    route_time = route_time_eval(route[0:break_point] + [depot], distance_matrix, TTW, ST)
                
                # print(">>> break point", break_point, route[0:break_point], route[break_point:])

                r_1 = route[0:break_point-1] + [depot]
                r_2 = [depot] + route[break_point-1:]

                routes.remove(route)
                routes.append(r_1)
                routes.append(r_2)

    solution_ = []
    for route in routes:
            solution_.extend(route)
    

    return solution_