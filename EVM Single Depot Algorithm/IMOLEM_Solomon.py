from classes import *
from functions import *

import pandas as pd
import re
from scipy.stats import pareto
import time


solomon_added_depots = {
    "RC101":[(8, 70), (7, 81)],
"RC102":[(18,43),(11, 46)],
"RC103":[(28, 74), (16, 14)],
"RC104":[(42, 78), (24, 85)],
"RC105":[(12, 25), (39, 40)],
"RC106":[(39, 38), (14, 15)],
"RC107":[(23, 37), (19, 31)],
"RC108":[(25, 44), (5, 30)],

"R101":[(18, 25), (41, 57)],
"R102":[(50, 53), (31, 41)],
"R103":[(29, 10), (34, 55)],
"R104":[(31, 52), (13, 46)],
"R105":[(42, 63), (63, 61)],
"R106":[(48, 5), (35, 31)],
"R107":[(14, 15), (41, 44)],
"R108":[(43, 43), (45, 49)],
"R109":[(44, 59), (48, 48)],
"R110":[(21, 37), (18, 63)],
"R111":[(30, 9), (8, 59)],
"R112":[(42, 27), (28, 64)],

"C101":[(31, 55), (40, 62)],
"C102":[(33, 60), (17, 66)],
"C103":[(34, 71), (32, 85)],
"C104":[(45, 71), (15, 85)],
"C105":[(45, 83), (29, 71)],
"C106":[(26, 67), (45, 51)],
"C107":[(33, 54), (32, 61)],
"C108":[(23, 64), (45, 69)]}


if __name__ == "__main__":
    selected_client = False
    
    client_number = 25
    
    Dataset = {"RC1":8, "R1":12, "C1":8}
    dataset_titles = {"RC1":"RC", "R1":"R", "C1":"C"}
    routes = {}
    times = {}
    
    for dataset in Dataset.keys():
        start_time = time.time()
        file_id = list(range(1, 1+Dataset[dataset]))
        for ind in file_id:
            ds_title = ""
            dataset_num = ind
            print(f"============ {dataset}_{dataset_num} ============ ")
            ## Data Preparation ##
            if dataset_num <=9:
                if dataset_titles[dataset] == "RC":
                    data_path = "./Datasets/Solomon/RC1/rc10"+str(dataset_num)+".txt"
                    ds_title = "RC10"+str(dataset_num)
                elif dataset_titles[dataset] == "C":
                    data_path = "./Datasets/Solomon/C1/c10"+str(dataset_num)+".txt"
                    ds_title = "C10"+str(dataset_num)
                elif dataset_titles[dataset] == "R":
                    data_path = "./Datasets/Solomon/R1/r10"+str(dataset_num)+".txt"
                    ds_title = "R10"+str(dataset_num)
            else:
                data_path = "./Datasets/Solomon/R1/r1"+str(dataset_num)+".txt"
                ds_title = "R1"+str(dataset_num)

            data_dataframe = pd.read_csv(data_path, delim_whitespace=True)

            print(data_dataframe)

            TW = data_dataframe['DUE_DATE']
            print("TW:\n",TW)
            print("Max DUE DATE =>",max(TW))
            time_bound = float(0.25*max(TW))
            print("Time Bound => 0.25", time_bound)

            coordinates = dict()
            time_windows = dict()
            demands = dict()
            service_times = dict()

            coordinates[0] = [data_dataframe["XCOORD."][0], list(data_dataframe["YCOORD."])[0]]
            time_windows[0] = [float(data_dataframe["READY_TIME"][0])-time_bound, float(list(data_dataframe["DUE_DATE"])[0])+time_bound]
            demands[0] = list(data_dataframe["DEMAND"])[0]
            service_times[0] = list(data_dataframe["SERVICE_TIME"])[0]

            shift = 1
            for d in solomon_added_depots[ds_title]:
                coords = d 
                coordinates[shift] = [d[0], d[1]]
                time_windows[shift] = [data_dataframe["READY_TIME"][0], list(data_dataframe["DUE_DATE"])[0]]
                demands[shift] = list(data_dataframe["DEMAND"])[0]
                service_times[shift] = list(data_dataframe["SERVICE_TIME"])[0]

            for item in list(dict(data_dataframe["XCOORD."]).keys())[1:client_number+1]:
                shift += 1
                coordinates[shift] = [data_dataframe["XCOORD."][item], list(data_dataframe["YCOORD."])[item]]
                time_windows[shift] = [data_dataframe["READY_TIME"][item], list(data_dataframe["DUE_DATE"])[item]]
                demands[shift] = list(data_dataframe["DEMAND"])[item]
                service_times[shift] = list(data_dataframe["SERVICE_TIME"])[item]

            print("Coordinates => ", coordinates, '\n')
            print("Time windows => ", time_windows, '\n')
            print("Demands => ", demands, '\n')
            print("Service Times => ", service_times, '\n')

            distance_matrix = compute_distances(list(coordinates.values()))
            print("Distance Matrix Shape=> ", distance_matrix.shape)

            #################################################################
            #################################################################
            ## Solomon Applied ##
            iteration = 1
            population_size = 100
            n_generation = 200
            length = len(coordinates)-1
            vehicle_capacity = 200
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
            if not len(P):
                print("\nZero Population\n\n")
                continue
            
            Q = []
            clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
            for iter in range(iteration):
                print(f"\n ========= Iteration {iter} =========== \n")
                obj_dict_Q = dict()
                for individual in Q:
                    obj_distance,obj_M, obj_subroutes = RSM(individual, distance_matrix, demands, time_windows, vehicle_capacity,service_times, B, m1, m2)

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

                    obj_dict_P[P.index(individual)] = objective_values


                pt_P_values = pareto_frontier_multi(np.array(list(obj_dict_P.values())))
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

            print(f"Dataset {dataset}, {dataset_num}, {client_number}")
            x = float(np.mean(results, axis=0)[0])
            y = float(np.mean(results, axis=0)[1])
            z = float(np.mean(results, axis=0)[2])
            print("Mean Distance =>", x)
            print("Mean M Values =>", y)
            print("Mean N Subroutes =>", z)
            k = str(f"Dataset {dataset}, {dataset_num}, {client_number}")
            routes[k] = [x, y, z]
            finish_time = time.time()
            total_time = finish_time-start_time
            times[k] = total_time
    
    for item in routes:
        print(item, " => ", routes[item])
        print(item, " => ", times[item])
