from classes import *
from functions import *
from utils import *
import pandas as pd
import re
from scipy.stats import pareto
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import time


if __name__ == "__main__":
    Total_Solution = dict()
    Total_Distances = dict()
    RunTime = dict()
    No_solutions = dict()
    vehicle_nums = dict()

    dataset_range = list(range(11, 25))
    codes = ['a', 'b']

    for code in codes:
       for dataset_num in dataset_range:
        ## Data Preparation ##
        data_path = f"./Dataset/Public/vidal-al-2013-mdvrptw/pr{dataset_num}{code}.txt"
        ds_title = f"Vidal_pr{dataset_num}{code}"

        print(f"\n=========== {ds_title} ===========\n")

        data_df, depot_df, vehicles, data_conf = reading_vidal_ds(data_path)
        depot_num = len(depot_df)

        data_df['XCOORD.'] = (data_df["XCOORD."]+100)/200
        data_df['YCOORD.'] = (data_df["YCOORD."]+100)/200
        data_df['DEMAND'] = data_df['DEMAND']/50 #max(data_df['DEMAND'])
        data_df['SERVICE_TIME'] = data_df['SERVICE_TIME']/50
        data_df['READY_TIME'] = data_df['READY_TIME']/1000
        data_df['DUE_DATE'] = data_df['DUE_DATE']/1000

        depot_df['XCOORD.'] = (depot_df["XCOORD."]+100)/200
        depot_df['YCOORD.'] = (depot_df["YCOORD."]+100)/200
        depot_df['DUE_DATE'] = depot_df['DUE_DATE']/1000

        depot_num = len(depot_df)
        client_number = len(data_df)

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
            index += 1

        for item in list(dict(data_df["XCOORD."]).keys()):
            coordinates[index] = [data_df["XCOORD."][item], list(data_df["YCOORD."])[item]]
            time_windows[index] = [data_df["READY_TIME"][item], list(data_df["DUE_DATE"])[item]]
            demands[index] = list(data_df["DEMAND"])[item]
            service_times[index] = list(data_df["SERVICE_TIME"])[item]
            index += 1

        start_time = time.time()
        distance_matrix = compute_distances(list(coordinates.values()))
        print("Distance Matrix Shape=> ", distance_matrix.shape)
        print("Total Client Number:", client_number,"\n=====================\n")

        cluster = dict()
        for i in range(depot_num):
            if i not in cluster.keys():
                cluster[i] = []
        
        for i in range(depot_num, client_number+depot_num):
            rand_depot = random.sample(list(cluster.keys()), k=1)[0]
            cluster[rand_depot].append(i)

        #print(cluster)
        #################################################################
        #################################################################
        iteration = 50
        population_size = 100 #2*client_number
        n_generation = 200 #2*client_number 
        length = client_number 
        vehicle_capacity = 5
        H_group_rate = 0.3
        L_group_rate = 0.3

        search_rate = 0.4
        swapping_rate = 0.5
        merging_rate = 0.5
        shuffling_rate =  0.3

        # B= 0.8*time_windows[0][1]
        B= 0.8*20 ## max route time
        m1 = 10
        m2 = 20

        H_group = []
        L_group = []

        populations = dict()
        Population = dict()
        vehicle_numbers = dict()
        pop_lengths = []
        for i in cluster.keys():
            POP = population(population_size, len(cluster[i]), cluster[i], i, len(cluster[i]), demands, vehicle_capacity, time_windows, distance_matrix, B, initialize=True)

            max_rep = 1000
            rep = 0
            while not len(POP.population) and rep <= max_rep:
                rep += 1
                if rep%100==0:
                    print("POP in progress", rep)
                POP = population(population_size, len(cluster[i]), cluster[i], i, len(cluster[i]), demands, vehicle_capacity, time_windows, distance_matrix, B, initialize=True)
                print(f"{i} While P => {len(POP.population)}")
            populations[i] = POP.population
            Population[i] = POP
            vehicle_numbers[i] = POP.vehicle_number
            pop_lengths.append(len(POP.population))
            print(f"P => {len(populations[i])}")
            print(f"Depot Population {i} Created\n================\n")

        print("Populations Generated.")
##        if min(pop_lengths) == 0:
##            continue

        Q = dict()
        for depot in cluster.keys():
            Q[depot] = []
        
        clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
        for iter in tqdm(list(range(iteration)), desc="\nIterations: "):
            # print(f"\n========= Iteration {iter} =========== \n")
            for depot in cluster.keys():
                client_keys = cluster[depot]

                obj_dict_Q = dict()
                for individual in Q[depot]:
                    obj_distance,obj_M, obj_subroutes = RSM(individual, distance_matrix, demands, time_windows, vehicle_capacity, service_times, B, m1, m2, client_keys, depot)
                    objective_values = [obj_distance, obj_M, obj_subroutes]
                    obj_dict_Q[Q[depot].index(individual)] = objective_values

                Q[depot].extend(populations[depot])

                if len(obj_dict_Q):
                    obj_list = np.array(list(obj_dict_Q.values()))
                    pt_Q_values = pareto_frontier_multi(obj_list)

                    Q_sorted = []
                    for i in pt_Q_values:
                        index = list(obj_dict_Q.values()).index(list(i))
                        if index in range(len(populations[depot])):
                            Q_sorted.append(populations[depot][index])
                    Q[depot] = Q_sorted[:population_size]
                    # print("Q sorted =>", Q)
                
                obj_dict_P = dict()
                for individual in populations[depot]:
                    obj_distance, obj_M, obj_subroutes = RSM(individual, distance_matrix, demands, time_windows, vehicle_capacity,service_times, B, m1, m2, client_keys, depot)
                    objective_values = [obj_distance, obj_M, obj_subroutes]
                    obj_dict_P[populations[depot].index(individual)] = objective_values

                pt_P_values = pareto_frontier_multi(np.array(list(obj_dict_P.values())))

                # print("Pareto sort", len(pt_P_values), pt_P_values)
                # print("Pareto sort", len(pt_P_values))
                P_sorted = []
                for i in pt_P_values:
                    index = list(obj_dict_P.values()).index(list(i))
                    P_sorted.append(populations[depot][index])

                populations[depot] = P_sorted

            # print("P sorted => ",P)
            # print(len(P), int(H_group_rate*len(P)))
            H_group = populations[depot][0: int(H_group_rate*len(populations[depot]))]
            L_group = populations[depot][-int(L_group_rate*len(populations[depot])):]
            if len(H_group)==0 and len(populations[depot])>1:
                H_group = populations[depot][0:int(len(populations[depot])/2)]
                L_group = populations[depot][-int(len(populations[depot])/2):]


            if len(populations[depot])==1:
                H_group = populations[depot]
                L_group = []

            X = [item for item in H_group]
            Y = [[1] for _ in H_group]
            X +=[item for item in L_group]
            Y += [[0] for _ in L_group]

            # print("H_group", H_group)
            # print("L_group", L_group)
            clf.fit(X,Y)

            # print()
            indexes = ["Client "+str(index) for index in range(1,len(cluster[depot])+1)]
            text_representation = tree.export_text(clf, feature_names=indexes)
            # print(text_representation)

            new_generation = Population[depot].create(clf, n_generation)
            # print("New generation ", len(new_generation) , new_generation)
            # print("New generation ", len(new_generation))

            new_generation.extend(Q[depot][0:int(len(Q[depot])/2)])
            # print("New generation ", len(new_generation) , new_generation)
            # print("New generation ", len(new_generation))

            populations[depot] = new_generation
            Population[depot].population = populations[depot]
            for individual in populations[depot][int(len(populations[depot])/2):]:
                index = populations[depot].index(individual)
                populations[depot][index] = Population[depot].heuristic_operations(populations[depot][index], distance_matrix, swapping_rate, merging_rate, shuffling_rate)

        results = []
        solutions = dict()
        for depot in Q.keys():
            solutions[depot] = []
            for item in Q[depot]:
                dist, M, n = RSM(item, distance_matrix, demands, time_windows, vehicle_capacity,service_times, B, m1, m2, cluster[depot], depot)
                results.append([dist, M, n])
                item_dict = get_swap_dict(dict(enumerate(item)))
                routes = sub_route_identifier(item)
                s_route = []
                for route in routes:
                    r_list = []
                    for r in route:
                        r_list.append(cluster[depot][item_dict[r]])
                    s_route.append(r_list)
                solutions[depot].append(s_route)
                    
        solution_distances = dict()
        for depot in solutions:
            solution_distances[depot] = []
            depot_dist = []
            for sample in solutions[depot]:
                sample_dist = 0
                for route in sample:
                    # print("Route =>", [depot]+route+[depot])
                    route_dist = tour_distance([depot]+route+[depot], distance_matrix)
                    sample_dist += route_dist
                depot_dist.append(sample_dist)
            solution_distances[depot] = np.mean(depot_dist)

        finish_time = time.time()
        Total_Solution[ds_title] = solutions
        Total_Distances[ds_title] = np.sum(list(solution_distances.values()))
        RunTime[ds_title] = finish_time - start_time
        vehicle_nums[ds_title] = vehicle_numbers
 

    print("\n\n==========================\n")
    distances_sum = []
    runtime_sum = []
    for key in Total_Solution.keys():
        print(key, "Solution: ", Total_Solution[key])
        print(key, "Distances per Depot: ", Total_Distances[key])
        print(key, "Vehicle Numbers: ", vehicle_nums[key])
        distances_sum.append(Total_Distances[key])
        print("Run Time:" ,RunTime[key])
        runtime_sum.append(RunTime[key])
        print("==========================\n")

    print("Mean Total Sum Distances on the instances: ", np.mean(distances_sum))
    print("Mean Total Run Time on the instances (s): ", np.mean(runtime_sum))
    print("No Solution Cases =>", No_solutions.keys())
