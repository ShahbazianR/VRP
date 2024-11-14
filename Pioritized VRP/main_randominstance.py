from Reading_dataset import *
from model_functions import *
from scipy.spatial.distance import euclidean
from Algorithm import RL_Agent
from VNS import *
from tqdm import tqdm
import time

## Idea: Add Priority to customers => Assuming that there is a priority in serving customers
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


if __name__ == "__main__":
    Total_Solution = dict()
    Total_Distances = dict()
    RunTime = dict()

    val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_20_2024-04-07.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_50_2024-04-07.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_100_2024-04-07.pkl'
    validation_dataset = read_from_pickle(val_set_path)

    instances = []
    for x in validation_dataset.batch(1):
        depots, customers, demand, time_windows, service_times = x
        cust_vectors, depot_vectors = create_vectors(customers, depots, demand, time_windows)
        cust_vectors = tf.convert_to_tensor(cust_vectors, dtype=tf.float64)
        depot_vectors = tf.convert_to_tensor(depot_vectors, dtype=tf.float64)
        instances.append((cust_vectors, demand, time_windows, service_times, depot_vectors, customers, depots))

    for instance_id in tqdm(list(range(len(instances))[0:5]), desc="Instances: "):
        print(f"============= Instance {instance_id} Initiated =============")
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

        coordinates_customers = dict()
        time_windows_customers = dict()
        demands_customers = dict()
        service_times_customers = dict()
        customers_info = dict()

        coordinates_depots = dict()
        time_windows_depots = dict()

        total_coords = dict()

        index = 0
        for item in range(depot_num):
            coordinates_depots[index] = [depot_df["XCOORD."][item], list(depot_df["YCOORD."])[item]]
            time_windows_depots[index] = [depot_df["READY_TIME"][item], list(depot_df["DUE_DATE"])[item]]

            total_coords[index] = [depot_df["XCOORD."][item], list(depot_df["YCOORD."])[item]]
            index += 1

        for item in list(dict(data_df["XCOORD."]).keys()):
            coordinates_customers[index] = [data_df["XCOORD."][item], list(data_df["YCOORD."])[item]]
            time_windows_customers[index] = [data_df["READY_TIME"][item], list(data_df["DUE_DATE"])[item]]
            demands_customers[index] = list(data_df["DEMAND"])[item]
            service_times_customers[index] = list(data_df["SERVICE_TIME"])[item]

            total_coords[index] = [data_df["XCOORD."][item], list(data_df["YCOORD."])[item]]
            index += 1

        # print("\n ======== Total Coordinates ========")
        # print("Coordinates =>\n", total_coords, '\n')

        # print("\n ======== Depots ========")
        # print("Coordinates =>\n", coordinates_depots, '\n')

        # print("\n ======== Customers ========")
        # print("Coordinates =>\n", coordinates_customers, '\n')

        ######### DRL #########
        C = coordinates_customers
        D = coordinates_depots
        TW_C = time_windows_customers
        ServiceT_C = service_times_customers
        Demands = demands_customers
        Max_VC = 1
        Max_VRT = 20
        n_episodes = 200
        n_vehicles = 3
        n_timesteps = max(data_df['DUE_DATE']) + 100

        alpha = 0.1
        beta = 0.2
        dist_rate=1
        time_rate=1

        ########### Algorithms ##########
        start_time = time.time()

        distance_matrix = compute_distances(total_coords)
        # print("Distance Matrix Created")
        # print("================================================")

        # clusters_ = generating_clusters(C, D)
        # print("Clusters => ", clusters_)

        clusters = dict()
        for d in D:
            clusters[d] = list()

        Ds = dict()
        for c in C.keys():
            min_depot = (None, np.inf)
            for d in D.keys():
                if distance_matrix[(c,d)] < min_depot[1]:
                    min_depot = (d, distance_matrix[(c,d)])
            Ds[c] = min_depot

        for item in Ds:
            depot = Ds[item][0]
            cust = item
            clusters[depot].append(cust)
    
        print(clusters)

        agent, routes = RL_Agent(C, D, TW_C, Demands, Max_VC, clusters, ServiceT_C, n_vehicles, n_episodes, n_timesteps, \
                Q_update_frequency = 2, Target_update_freq = 4, route_max_time = Max_VRT, 
                alpha=alpha, beta=beta)
        
        print("\n\nInitial routes >>", routes)

        initial_solution = dict()
        for depot in routes.keys():
            initial_solution[depot] = []
            for route in routes[depot].values():
                for r in route:
                    initial_solution[depot].append([depot] + r + [depot])

        print("\n\nInitial Solution >>", initial_solution)
        
        
        ## Sample Initial Solution to Test the VNS Module
        #  initial_solution = {0: [[0, 7, 8, 9, 10, 6, 22, 0], [0, 12, 19, 14, 17, 20, 0]], 1: [[1, 11, 3, 1]], 2: [[2, 16, 13, 15, 23, 18, 2], [2, 21, 5, 4, 2]]}


        ############# VNS #############

        final_solution = dict()
        initial_distance = 0
        final_distance = 0
        for depot in initial_solution.keys():
            final_solution[depot] = []
            for route in initial_solution[depot]:
                init_dist = tour_distance(route, distance_matrix)
                initial_distance += init_dist
                # print("Initial Distance =>", init_dist) 
                best_tour, exploration_time, exploitation_time = vns(route, distance_matrix, TW_C, ServiceT_C, alpha, beta, dist_rate, time_rate, k_max=50, operator=two_opt)
                print(best_tour, exploitation_time, exploration_time)
                fin_dist  = tour_distance(best_tour, distance_matrix)
                # print("Final Distance =>", fin_dist)
                final_distance += fin_dist
                final_solution[depot].append(best_tour)
                # print("===============================")

        print("Initial Solution >>", initial_solution)
        print("Initial Distance >>", initial_distance)
        print()
        print("Final Solution >>", final_solution)
        print("Final Distance >>", final_distance)

        Total_Solution[instance_id] = final_solution
        Total_Distances[instance_id] = {"Initial Solution": np.sum(initial_distance),
                                         "Sum": np.sum(final_distance)}
        finish_time = time.time()
        RunTime[instance_id] = finish_time - start_time
        print(f"\n================ Evaluation For Instance {instance_id} Finished ======================\n")
    
    distances_sum = []
    runtime_sum = []
    for key in Total_Solution.keys():
        print(key, "Solution: ", Total_Solution[key])
        print(key, "Distances per Depot: ", Total_Distances[key])
        distances_sum.append(Total_Distances[key]["Sum"])
        print("Run Time:" ,RunTime[key])
        runtime_sum.append(RunTime[key])
        print("==========================\n")

    print("Mean Total Sum Distances on the instances: ", np.mean(distances_sum))
    print("Mean Total Run Time on the instances (s): ", np.mean(runtime_sum))