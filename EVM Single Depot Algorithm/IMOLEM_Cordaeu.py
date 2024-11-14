from classes import *
from functions import *
from utils import *

import pandas as pd
import re
from scipy.stats import pareto

if __name__ == "__main__":
    dataset_num = 1
    client_number = 16

    selected_client = False

    ## Data Preparation ##
    if dataset_num <=9:
        data_path = "./Data/cordeau2001-mdvrptw/pr0"+str(dataset_num)+".txt"
    else:
        data_path = "./Data/cordeau2001-mdvrptw/pr"+str(dataset_num)+".txt"

    data_df, depot_df, data_conf = reading_cordeu_ds(data_path)
    depot_num = len(depot_df)

    # if selected_client:
    #     file = open(chosen_clients_file, 'r')
    #     read = file.readlines()

    #     chosen_clients = dict()
    #     index = 1
    #     for item in read:
    #         ind_list = dict()
    #         ind_list[0] = 0
    #         id = 1
    #         temps = re.split('\n', item)[0]
    #         temps = re.split(', ', temps)
    #         for i in temps:
    #         ind_list[id] = int(i)
    #         id += 1
    #         chosen_clients[index] = ind_list
    #         index += 1

    #     print(chosen_clients[dataset_num])

    #     depot_num = 1
    #     evaluation_df = pd.DataFrame()
    #     evaluation_df = data_dataframe[0:depot_num]
    #     for index in list(chosen_clients[dataset_num].values()):
    #         evaluation_df = pd.concat([evaluation_df, data_dataframe[index-1:index]], ignore_index=True)

    #     data_dataframe = evaluation_df
    #     print(evaluation_df)

    coordinates = dict()
    time_windows = dict()
    demands = dict()
    service_times = dict()

    index = 0
    for item in range(depot_num):
        coordinates[index] = [depot_df["XCOORD."][item], list(depot_df["YCOORD."])[item]]
        time_windows[index] = [depot_df["READY_TIME"][item], list(depot_df["DUE_DATE"])[item]]
        demands[index] = list(depot_df["DEMAND"])[item]
        service_times[index] = list(depot_df["SERVICE_TIME"])[item]

    for item in list(dict(data_df["XCOORD."]).keys())[1:client_number+1]:
        coordinates[item] = [data_df["XCOORD."][item], list(data_df["YCOORD."])[item]]
        time_windows[item] = [data_df["READY_TIME"][item], list(data_df["DUE_DATE"])[item]]
        demands[item] = list(data_df["DEMAND"])[item]
        service_times[item] = list(data_df["SERVICE_TIME"])[item]

    print(coordinates)
    print(time_windows)
    print(demands)
    print(service_times)

    distance_matrix = compute_distances(list(coordinates.values()))
    print("Distance Matrix Shape=> ", distance_matrix.shape)

    #################################################################
    #################################################################
    ## Solomon Applied ##
    iteration = 10
    population_size = 100
    n_generation = 200
    length = len(coordinates)-1
    vehicle_capacity = 100
    H_group_rate = 0.3
    L_group_rate = 0.3

    search_rate = 0.4
    swapping_rate = 0.5
    merging_rate = 0.5
    shuffling_rate =  0.3

    B= 0.8*time_windows[0][1]
    m1 = 10
    m2 = 20

    H_group = []
    L_group = []

    Population = population(population_size, length, client_number, demands, vehicle_capacity, time_windows, distance_matrix, initialize=True)
    P = Population.population
    print(f"P => {len(P)}",P)

    Q = []
    clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    for iter in range(iteration):
        print(f"\n ========= Iteration {iter} =========== \n")
        obj_dict_Q = dict()
        for individual in Q:
            obj_distance,obj_M, obj_subroutes = RSM(individual, distance_matrix, demands, time_windows, vehicle_capacity,service_times, B, m1, m2)
            # dist = route_distance(individual, distance_matrix)
            # n_subroutes = len(sub_route_identifier(individual))
            # obj_distance = dist
            # obj_M = Population.driver_renumeration(individual, distance_matrix, service_times, B, m1, m2)
            # objective_values = [obj_distance, obj_M, n_subroutes]

            objective_values = [obj_distance, obj_M, obj_subroutes]
            obj_dict_Q[Q.index(individual)] = objective_values


        Q.extend(P)
        if len(obj_dict_Q):
            obj_list = np.array(list(obj_dict_Q.values()))
            pt_Q_values = pareto_frontier_multi(obj_list)

            Q_sorted = []
            for i in pt_Q_values:
                index = list(obj_dict_Q.values()).index(list(i))
                Q_sorted.append(P[index])
            Q = Q_sorted[:population_size]
                # print("Q sorted =>", Q)

        obj_dict_P = dict()
        for individual in P:
            obj_distance, obj_M, obj_subroutes = RSM(individual, distance_matrix, demands, time_windows, vehicle_capacity,service_times, B, m1, m2)
            objective_values = [obj_distance, obj_M, obj_subroutes]

            # dist = route_distance(individual, distance_matrix)
            # n_subroutes = len(sub_route_identifier(individual))
            # obj_distance = dist
            # obj_M = driver_renumeration(individual, distance_matrix, service_times, B, m1, m2, [])
            # objective_values = [obj_distance, obj_M, n_subroutes]

            obj_dict_P[P.index(individual)] = objective_values


        pt_P_values = pareto_frontier_multi(np.array(list(obj_dict_P.values())))
        # print("Pareto sort", len(pt_P_values), pt_P_values)
        print("Pareto sort", len(pt_P_values))
        P_sorted = []
        for i in pt_P_values:
            index = list(obj_dict_P.values()).index(list(i))
            P_sorted.append(P[index])

        P = P_sorted
        # print("P sorted => ",P)
        print(len(P), int(H_group_rate*len(P)))
        H_group = P[0: int(H_group_rate*len(P))]
        L_group = P[-int(L_group_rate*len(P)):]
        if len(H_group)==0 and len(P)>1:
            H_group = P[0:int(len(P)/2)]
            L_group = P[-int(len(P)/2):]

        if len(P)==1:
            H_group = P
            L_group = []

        X = [item for item in H_group]
        Y = [[1] for item in H_group]
        X +=[item for item in L_group]
        Y += [[0] for item in L_group]

        print("H_group", H_group)
        print("L_group", L_group)
        clf.fit(X,Y)

        print()
        indexes = ["Client "+str(index) for index in range(1,length+1)]
        text_representation = tree.export_text(clf, feature_names=indexes)
        print(text_representation)

        new_generation = Population.create(clf, n_generation)
        # print("New generation ", len(new_generation) , new_generation)
        print("New generation ", len(new_generation))
        new_generation.extend(Q[0:int(len(Q)/2)])
        # print("New generation ", len(new_generation) , new_generation)
        print("New generation ", len(new_generation))
        P = new_generation
        Population.population = P
        for individual in P[int(len(P)/2):]:
            index = P.index(individual)
            P[index] = Population.heuristic_operations(P[index], distance_matrix, swapping_rate, merging_rate, shuffling_rate)

        # print("P", len(P) , P)
        print("P", len(P))

    print("Q is :")
    print(Q)


    results = []
    for item in Q:
        dist, M, n = RSM(item, distance_matrix, demands, time_windows, vehicle_capacity,service_times, B, m1, m2)
        results.append([dist, M, n])

    print(results)

    print(f"Dataset Cordaeu pr_{dataset_num}, {client_number}")
    print(np.mean(results, axis=0)[0])
    print(np.mean(results, axis=0)[1])
    print(np.mean(results, axis=0)[2])