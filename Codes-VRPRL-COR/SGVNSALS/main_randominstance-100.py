from functions import *
from model_functions import *
from scipy.spatial.distance import euclidean
import time
from tqdm import tqdm

# Customers = [[0.81156135, 0.51285875],
#   [0.5158731 , 0.1984117 ],
#   [0.10018826, 0.6977637 ],
#   [0.8432797 , 0.8220134 ],
#   [0.60395014, 0.7521423 ],
#   [0.22587848, 0.9106029 ],
#   [0.1619848 , 0.73984337],
#   [0.8821931 , 0.98377   ],
#   [0.8240005 , 0.09986246],
#   [0.96484745, 0.86558187],
#   [0.60768855, 0.42057073],
#   [0.22018921, 0.7029598 ],
#   [0.24856472, 0.1793282 ],
#   [0.66513515, 0.03073382],
#   [0.18058157, 0.38004386],
#   [0.08649564, 0.10784924],
#   [0.30336654, 0.75960886],
#   [0.08908963, 0.8686968 ],
#   [0.63589454, 0.15933275],
#   [0.16827607, 0.9322336 ],
#   [0.4741609 , 0.2654822 ]]

# Depots = [[0.7716589 , 0.807168],
#   [0.93755794, 0.49416363],
#   [0.6536195 , 0.31947935]]

# Demands = [0.13333334, 0.16666667, 0.23333333, 0.03333334, 0.3       , 0.23333333, 
#            0.03333334, 0.3       , 0.13333334, 0.16666667, 0.16666667, 0.23333333,
#            0.2       , 0.1       , 0.16666667, 0.26666668, 0.3       , 0.06666667,
#            0.2       , 0.06666667, 0.13333334]

# TimesWindows = [[0.5 , 0.95],
#                 [0.15, 0.75],
#                 [0.6 , 0.9 ],
#                 [0.2 , 0.3 ],
#                 [0.7 , 0.95],
#                 [0.15, 0.6 ],
#                 [0.35, 0.85],
#                 [0.6 , 0.7 ],
#                 [0.15, 0.65],
#                 [0.3 , 0.45],
#                 [0.9 , 0.95],
#                 [0.05, 0.45],
#                 [0.4 , 0.7 ],
#                 [0.2 , 0.45],
#                 [0.55, 0.55],
#                 [0.15, 0.2 ],
#                 [0.55, 0.95],
#                 [0.85, 0.85],
#                 [0.4 , 0.55],
#                 [0.15, 0.3 ],
#                 [0.05, 0.75]]
######################################
# Customers = [[0.19668865, 0.03486848],
#   [0.14022255, 0.2044003 ],
#   [0.21368277, 0.8652357 ],
#   [0.9702513 , 0.17152667],
#   [0.8174552 , 0.7812284 ],
#   [0.66081154, 0.8857814 ],
#   [0.7392056 , 0.59419096],
#   [0.63479936, 0.5488807 ],
#   [0.7570567 , 0.40136373],
#   [0.39033258, 0.90448713],
#   [0.05026674, 0.88066626],
#   [0.82142234, 0.59845936],
#   [0.30969203, 0.99131477],
#   [0.8076122 , 0.13640857],
#   [0.15419364, 0.9787388 ],
#   [0.569754  , 0.3073939 ],
#   [0.91505325, 0.77885747],
#   [0.01658893, 0.84748614],
#   [0.60924304, 0.59881294],
#   [0.23199046, 0.74871063],
#   [0.071105  , 0.7851796 ]]

# Depots = [[0.44433177, 0.16821373],
#   [0.6199864 , 0.17532861],
#   [0.79592717, 0.5404875 ]]

# Demands=[0.23333333, 0.2       , 0.3       , 0.03333334, 0.03333334, 0.26666668
#  , 0.13333334, 0.23333333, 0.16666667, 0.16666667, 0.1       , 0.13333334
#  , 0.3       , 0.26666668, 0.26666668, 0.3       , 0.06666667, 0.2
#  , 0.06666667, 0.1       , 0.3       ]

# TimesWindows = [[0.05, 0.3 ],
#   [0.25, 0.5 ],
#   [0.35, 0.55],
#   [0.3 , 0.45],
#   [0.05, 0.7 ],
#   [0.7 , 0.95],
#   [0.35, 0.8 ],
#   [0.8 , 0.95],
#   [0.2 , 0.45],
#   [0.35, 0.85],
#   [0.15, 0.2 ],
#   [0.1 , 0.2 ],
#   [0.45, 0.65],
#   [0.05, 0.8 ],
#   [0.35, 0.55],
#   [0.1 , 0.1 ],
#   [0.2 , 0.3 ],
#   [0.6 , 0.65],
#   [0.75, 0.95],
#   [0.65, 0.95],
#   [0.1 , 0.5 ]]


if __name__ == "__main__":
    routes = dict()
    distances = dict()
    vehicle_count = dict()
    RunTime = dict()

    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_20_2024-04-07.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_50_2024-04-07.pkl'
    val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_100_2024-04-07.pkl'
    validation_dataset = read_from_pickle(val_set_path)

    n_depots = 3

    instances = []
    for x in validation_dataset.batch(1):
        depots, customers, demand, time_windows, service_times = x
        cust_vectors, depot_vectors = create_vectors(customers, depots, demand, time_windows)
        cust_vectors = tf.convert_to_tensor(cust_vectors, dtype=tf.float64)
        depot_vectors = tf.convert_to_tensor(depot_vectors, dtype=tf.float64)
        instances.append((cust_vectors, demand, time_windows, service_times, depot_vectors, customers, depots))

    print(len(instances))
    
    for instance_id in tqdm(list(range(len(instances))), desc="Instances: "):
        print(f"/n============= Instance {instance_id} =============")
        _, demands, time_windows, service_times, _, coords_custs, coords_depot = instances[instance_id]

        Demands = demands.numpy()[0]
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

##        print("Customers =>", Customers)
##        print("Demands =>", Demands)
##        print("Depots =>", Depots)
##        print("TimeWindows =>", TimesWindows)

        total_coords = dict()

        index = 0
        for item in range(n_depots):
            total_coords[index] = [coords_depot[0][item][0].numpy(), coords_depot[0][item][1].numpy()]
            index += 1

        for item in range(len(coords_custs[0])):
            total_coords[index] = [coords_custs[0][item][0].numpy(), coords_custs[0][item][1].numpy()]
            index += 1

        # print("\n ======== Total Coordinates ========")
        # print("Coordinates =>\n", total_coords, '\n')

        ## Data Preparation ##
        columns=['CUST_NO.', 'XCOORD.', 'YCOORD.', 'SERVICE_TIME', 'DEMAND', 'READY_TIME', 'DUE_DATE']

        print(Depots)
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

        coordinates_customers = dict()
        time_windows_customers = dict()
        demands_customers = dict()
        service_times_customers = dict()
        customers_info = dict()

        coordinates_depots = dict()
        time_windows_depots = dict()

        index = 0
        for item in range(depot_num):
            coordinates_depots[index] = [depot_df["XCOORD."][item], list(depot_df["YCOORD."])[item]]
            time_windows_depots[index] = [depot_df["READY_TIME"][item], list(depot_df["DUE_DATE"])[item]]
            index += 1

        for item in list(dict(data_df["XCOORD."]).keys()):
            coordinates_customers[index] = [data_df["XCOORD."][item], list(data_df["YCOORD."])[item]]
            time_windows_customers[index] = [data_df["READY_TIME"][item], list(data_df["DUE_DATE"])[item]]
            demands_customers[index] = list(data_df["DEMAND"])[item]
            service_times_customers[index] = list(data_df["SERVICE_TIME"])[item]
            index += 1

        # print("\n ======== Depots ========")
        # print("Coordinates =>\n", coordinates_depots, '\n')
        # # print("Time Windows =>\n",time_windows_depots)

        # print("\n ======== Customers ========")
        # print("Coordinates =>\n", coordinates_customers, '\n')
        # print("Time Windows =>\n",time_windows_customers, '\n')
        # print("Demands =>\n",demands_customers, '\n')
        # print("Service Times =>\n",service_times_customers)

        start_time = time.time()
        distance_matrix = compute_distances(list(coordinates_customers.values()))
        # print("Distance Matrix Shape=> ", distance_matrix.shape)

        print("\n======== SGVNSALS Algorithm ========\n")
        C = coordinates_customers
        D = coordinates_depots
        iterMax = 50
        maxTime = 20
        maxLevel = 5
        pLS1 = 0.3
        pLS2 = 0.3
        als = True
        TW_C = time_windows_customers
        ServiceT_C = service_times_customers
        Demands = demands_customers
        Vehicle_info = {'max_T': [maxTime], 'max_load': [1]}
        lamda = 0.6 ## From Empirical Tests

        final_solution = SGVNSALS(C, D, iterMax, maxTime, maxLevel, pLS1, pLS2, als, TW_C, ServiceT_C, Demands, Vehicle_info, lamda)
        
        print("\n Final Solution: ", final_solution)
        served = []
        seen_clients = []
        for key in final_solution:
            for r in final_solution[key]:
                served.extend(r)
                seen_clients.extend(set(r))
        
        print("\nServed Clients")
        print(sorted(served))
        print(len(served), len(set(served)))
        not_seen_count = 0
        for item in list(C.keys()):
            if item not in seen_clients:
                print("Not Seen =>", item)
                not_seen_count += 1

        print("\nCalculating the Distances in the Solution:")
        Export_Solution = {}
        evaluation_distances = []
        routes_num = 0
        for key in final_solution.keys():
            Export_Solution[key] = []
            dist = 0
            r = final_solution[key]
            # print(f"Depot: {key}, Total Associated Routes: {r}")
            for route_ in r:
                routes_num += 1
                route = [key] + route_ + [key]

                Export_Solution[key].append(route)
                for i in range(0, len(route)-1):
                    euclidean_distance = 0
                    if route[i] in D.keys() and route[i+1] in C.keys():
                        a = (D[route[i]][0], D[route[i]][1])
                        b = (C[route[i+1]][0], C[route[i+1]][1])
                        euclidean_distance = euclidean(a,b)
                    elif route[i] in C.keys() and route[i+1] in D.keys():
                        a = (C[route[i]][0], C[route[i]][1])
                        b = (D[route[i+1]][0], D[route[i+1]][1])
                        euclidean_distance = euclidean(a,b)
                    elif route[i] in C.keys() and route[i+1] in C.keys():
                        a = (C[route[i]][0], C[route[i]][1])
                        b = (C[route[i+1]][0], C[route[i+1]][1])
                        euclidean_distance = euclidean(a,b)
                    dist += euclidean_distance
            #     print(f"Depot: {key}, Route: {route}, Distance so far: {dist}")
            # print(f"Distance for All the Routes Associated with the Depot {key}: ", dist)
            evaluation_distances.append(dist)

        print("Number of vehicles =>", routes_num)
        print(evaluation_distances)
        print(f"Mean distance in evaluation {client_number} data {np.mean(evaluation_distances)}")
        print(f"Sum distance in evaluation {client_number} data {np.sum(evaluation_distances)}")
        print(f"Extracted Solution:\n {Export_Solution}")

        routes[instance_id] = Export_Solution
        distances[instance_id] = {"Mean": np.mean(evaluation_distances), "Sum": np.sum(evaluation_distances),
                                  'Not_seen':not_seen_count}
        vehicle_count[instance_id] = routes_num
        finish_time = time.time()
        RunTime[instance_id] = finish_time - start_time
        print(f"\n================ Evaluation For Instance {instance_id} Finished ======================\n")


    distances_sum = []
    runtime_sum = []
    for key in routes.keys():
        print(key, "Solution: ", routes[key])
        print(key, "Distances per Depot: ", distances[key])
        distances_sum.append(distances[key]["Sum"])
        print(key, "Vehicle Number: ", vehicle_count[key])
        print("Run Time:" ,RunTime[key])
        runtime_sum.append(RunTime[key])
        print("==========================\n")

    
    print("Mean Total Sum Distances on the instances: ", np.mean(distances_sum))
    print("Mean Total Run Time on the instances (s): ", np.mean(runtime_sum))
