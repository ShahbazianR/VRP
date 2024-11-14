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

    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_20_2024-04-07.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_50_2024-04-07.pkl'
    val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_100_2024-04-07.pkl'
    validation_dataset = read_from_pickle(val_set_path)

    print("Validation Dataset", val_set_path)

    instances = []
    for x in validation_dataset.batch(1):
        depots, customers, demand, time_windows, service_times = x
        cust_vectors, depot_vectors = create_vectors(customers, depots, demand, time_windows)
        cust_vectors = tf.convert_to_tensor(cust_vectors, dtype=tf.float64)
        depot_vectors = tf.convert_to_tensor(depot_vectors, dtype=tf.float64)
        instances.append((cust_vectors, demand, time_windows, service_times, depot_vectors, customers, depots))

    for instance_id in tqdm(list(range(len(instances))[0:10]), desc="Instances: "):
        print(f"\n============= Instance {instance_id} Initiated =============")
        _, demand, time_windows, service_times, _, coords_custs, coords_depot = instances[instance_id]

        Demands = demand.numpy()[0]
        Services = service_times.numpy()[0]

        Customers = []
        Depots = []
        TimesWindows = []
        for item in coords_custs.numpy()[0]:
            Customers.append([item[0], item[1]])

        for item in coords_depot.numpy()[0]:
            Depots.append([item[0], item[1]])
            
        for item in time_windows.numpy()[0]:
            TimesWindows.append([item[0], item[1]])

        # print("Customers =>", Customers)
        # print("Demands =>", Demands)
        # print("Depots =>", Depots)
        # print("TimeWindows =>", TimesWindows)

        total_coords = dict()

        index = 0
        for item in range(len(Depots)):
            total_coords[index] = [coords_depot[0][item][0].numpy(), coords_depot[0][item][1].numpy()]
            index += 1

        for item in range(len(coords_custs[0])):
            total_coords[index] = [coords_custs[0][item][0].numpy(), coords_custs[0][item][1].numpy()]
            index += 1

        ## Data Preparation ##
        columns=['CUST_NO.', 'XCOORD.', 'YCOORD.', 'SERVICE_TIME', 'DEMAND', 'READY_TIME', 'DUE_DATE']

        depots = copy.deepcopy(Depots)
        for ind in range(len(depots)):
            if ind < len(Depots):
                depots[ind].insert(0, ind)
                depots[ind].extend([0, 0, 0, 1])

        nodes = Customers
        for ind in range(len(nodes)):
            nodes[ind].insert(0, ind)
            nodes[ind].extend([Services[ind], Demands[ind], TimesWindows[ind][0], TimesWindows[ind][1]])

        data_df = pd.DataFrame(nodes, columns=columns)
        depot_df = pd.DataFrame(depots, columns=columns)
        # print("Customers:\n", data_df)
        # print()
        # print("Depots:\n", depot_df)
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

        # print(coordinates.keys())
        # print(time_windows)
        # print(demands)
        # print(service_times)

        start_time = time.time()
        distance_matrix = compute_distances(list(coordinates.values()))
        print("Distance Matrix Shape=> ", distance_matrix.shape)
        print("Client Number:", client_number)

        #################################################################
        #################################################################
        iteration = 50
        population_size = 100
        n_generation = 200
        length = len(total_coords)-1
        vehicle_capacity = 1
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

        Population = population(population_size, length, client_number, demands, vehicle_capacity, time_windows, distance_matrix, B, initialize=True)
        P = Population.population
        # print(f"P => {len(P)}",P)
        print(f"P => {len(P)}")

        if len(P) == 0:
            No_solutions[instance_id] = 1
            print("No Solutions")
            continue    


        Q = []
        clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
        for iter in (range(iteration)):
            print(f"========= Iteration {iter} ===========")
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
            # print("Pareto sort", len(pt_P_values))
            P_sorted = []
            for i in pt_P_values:
                index = list(obj_dict_P.values()).index(list(i))
                P_sorted.append(P[index])

            P = P_sorted
            # print("P sorted => ",P)
            # print(len(P), int(H_group_rate*len(P)))
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

            # print("H_group", H_group)
            # print("L_group", L_group)
            clf.fit(X,Y)

            # print()
            indexes = ["Client "+str(index) for index in range(1,length+1)]
            text_representation = tree.export_text(clf, feature_names=indexes)
            # print(text_representation)

            new_generation = Population.create(clf, n_generation)
            # print("New generation ", len(new_generation) , new_generation)
            # print("New generation ", len(new_generation))
            new_generation.extend(Q[0:int(len(Q)/2)])
            # print("New generation ", len(new_generation) , new_generation)
            # print("New generation ", len(new_generation))
            P = new_generation
            Population.population = P
            for individual in P[int(len(P)/2):]:
                index = P.index(individual)
                P[index] = Population.heuristic_operations(P[index], distance_matrix, swapping_rate, merging_rate, shuffling_rate)

            # print("P", len(P) , P)
            # print("P", len(P))

        print("Q is :")
        print(Q)

        results = []
        for item in Q:
            dist, M, n = RSM(item, distance_matrix, demands, time_windows, vehicle_capacity,service_times, B, m1, m2)
            results.append([dist, M, n])

        print(results)
        finish_time = time.time()
        Total_Solution[instance_id] = Q
        Total_Distances[instance_id] = {"Distance":np.mean(results, axis=0)[0], 
                                        "M": np.mean(results, axis=0)[1],
                                        "n": np.mean(results, axis=0)[2]}
        RunTime[instance_id] = finish_time - start_time

    print()
    distances_sum = []
    runtime_sum = []
    for key in Total_Solution.keys():
        print(key, "Solution: ", Total_Solution[key])
        print(key, "Distances per Depot: ", Total_Distances[key])
        distances_sum.append(Total_Distances[key]["Distance"])
        print("Run Time:" ,RunTime[key])
        runtime_sum.append(RunTime[key])
        print("==========================\n")

    print("Mean Total Sum Distances on the instances: ", np.mean(distances_sum))
    print("Mean Total Run Time on the instances (s): ", np.mean(runtime_sum))
    print("No Solution Cases =>", No_solutions.keys())
