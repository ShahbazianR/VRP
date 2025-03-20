from Reading_dataset import *
from model_functions import *
from scipy.spatial.distance import euclidean
from Algorithm import RL_Agent
from VNS import *
from tqdm import tqdm
import time


random_ds = ['UValidation_dataset_2D_VRP_20_2024-04-14',
            'UValidation_dataset_2D_VRP_50_2024-04-14',
            'UValidation_dataset_2D_VRP_100_2024-04-14',

            'UValidation_dataset_VRP_20_2024-04-07',
            'UValidation_dataset_VRP_50_2024-04-07',
            'UValidation_dataset_VRP_100_2024-04-07',

            'UValidation_dataset_4D_VRP_20_2024-04-14',
            'UValidation_dataset_4D_VRP_50_2024-04-14',
            'UValidation_dataset_4D_VRP_100_2024-04-14',

            'UValidation_dataset_5D_VRP_20_2024-04-14',
            'UValidation_dataset_5D_VRP_50_2024-04-14',
            'UValidation_dataset_5D_VRP_100_2024-04-14']



if __name__ == "__main__":
    data_dir = 'Dataset/Random_Converted'


    for ds_name in random_ds:

        Total_Solution = dict()
        Total_Distances = dict()
        RunTime = dict()
        results = []


        ## If Tensorflow is installed on the machine, you can read the datasets using the following code.
        ## The following loads the data using pickle. Note that when using this method, the data is returned as tensor
        ## and you need to modify the code below, converting tensors to numpy arrays when reading the data.
        # val_set_path = f"./Dataset/Random/Uniform Distribution/{ds_name}.pkl" 
        # validation_dataset = read_from_pickle(val_set_path)
        # instances = []
        # for x in validation_dataset.batch(1):
        #     depots, customers, demand, time_windows, service_times = x
        #     instances.append((demand, time_windows, service_times, customers, depots))


        ## If Tensorflow is not installed on the machine, please use the following code.
        ## This utilizes the numpy version of the same datasets.
        depots = np.load(f'{data_dir}/depots_{ds_name}.npy')
        customers = np.load(f'{data_dir}/customers_{ds_name}.npy')
        demand = np.load(f'{data_dir}/demand_{ds_name}.npy')
        time_windows = np.load(f'{data_dir}/time_windows_{ds_name}.npy')
        service_times = np.load(f'{data_dir}/service_times_{ds_name}.npy')

        instances = []
        for ind in range(depots.shape[0]):
            inst_depots = depots[ind]
            inst_customers = customers[ind]
            inst_demand = demand[ind]
            inst_time_windows = time_windows[ind]
            inst_service_times = service_times[ind]
            
            instances.append((inst_demand, inst_time_windows, inst_service_times, inst_customers, inst_depots))

        print(f"\n============= Dataset {ds_name} =============")
        for instance_id in tqdm(list(range(len(instances))), desc="Instances: "):
            print(f"\n============= Instance {instance_id} Initiated =============")
            demands, time_windows, service_times, coords_custs, coords_depot = instances[instance_id]

            Demands = demands
            Services = service_times

            Customers = []
            Depots = []
            TimesWindows = []
            for item in coords_custs:
                Customers.append([item[0], item[1]])

            for item in coords_depot:
                Depots.append([item[0], item[1]])
                
            for item in time_windows:
                TimesWindows.append([item[0], item[1]])

            # print("Customers =>", Customers)
            # print("Demands =>", Demands)
            # print("Depots =>", Depots)
            # print("TimeWindows =>", TimesWindows)

            total_coords = dict()

            index = 0
            for item in range(len(Depots)):
                total_coords[index] = [coords_depot[item][0], coords_depot[item][1]]
                index += 1

            for item in range(len(coords_custs[0])):
                total_coords[index] = [coords_custs[item][0], coords_custs[item][1]]
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
            n_vehicles = depot_num
            n_timesteps = max(data_df['DUE_DATE']) + 100

            alpha = 0.1
            beta = 0.2
            dist_rate=1
            time_rate=1

            ########### Algorithms ##########
            start_time = time.time()

            distance_matrix = compute_distances(total_coords)
            print("Distance Matrix Created")
            print("================================================")

            clusters_ = generating_clusters(C, D)
            print("Clusters => ", clusters_)

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
        
            print('Clusters:', clusters)

            agent, routes = RL_Agent(C, D, TW_C, Demands, Max_VC, clusters, ServiceT_C, n_vehicles, n_episodes, n_timesteps, \
                    Q_update_frequency = 2, Target_update_freq = 4, route_max_time = Max_VRT, 
                    alpha=alpha, beta=beta)
            
            print("\n\nInitial routes >>", routes)

            initial_solution = dict()
            for depot in routes.keys():
                initial_solution[depot] = []
                for route in routes[depot].values():
                    for r in route:
                        if len(r):
                            initial_solution[depot].extend([depot] + r + [depot])

            print("\n\nInitial Solution >>", initial_solution)

            ############# VNS #############

            final_solution = dict()
            initial_distance = 0
            final_distance = 0
            for depot in initial_solution.keys():
                final_solution[depot] = []
                route = initial_solution[depot]
                if len(route)>2:
                    init_dist = tour_distance(route, distance_matrix)
                    initial_distance += init_dist
                    # print("Initial Distance =>", init_dist) 
                    best_tour, exploration_time, exploitation_time = vns(route, depot, Demands, Max_VC, distance_matrix, TW_C, ServiceT_C, alpha, beta, dist_rate, time_rate, k_max=50, operator=two_opt)
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
            results.append({"DS":instance_id, "Initial_Solution": initial_solution, 
                            "Final_Solution": final_solution, 
                            "Initial_Dist": np.sum(initial_distance),
                            "Final_Dist": np.sum(final_distance),
                            "Time": RunTime[instance_id]})
            print(f"\n================ Evaluation For Instance {instance_id} Finished ======================\n")
        
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'./PAVNS_Random__{ds_name}.csv')

        file = open(f'Random_PAVNS_Log_{ds_name}.txt', 'w')
        distances_sum = []
        runtime_sum = []
        for key in Total_Solution.keys():
            print(key, "Solution: ", Total_Solution[key])
            print(key, "Distance:", Total_Distances[key])
            distances_sum.append(Total_Distances[key]["Sum"])
            print("Run Time:" ,RunTime[key])
            runtime_sum.append(RunTime[key])
            print("==========================\n")

            file.write(f'{key} >> Solution: {Total_Solution[key]}\n')
            file.write(f'{key} >> Distance: {Total_Distances[key]["Sum"]}\n')
            file.write(f'{key} >> Run Time: {RunTime[key]}\n')
            file.write("==========================\n")

        print("Mean Total Sum Distances on the instances: ", np.mean(distances_sum))
        print("Mean Total Run Time on the instances (s): ", np.mean(runtime_sum))

        file.write(f"Mean Total Sum Distances on the instances: {np.mean(distances_sum)}\n")
        file.write(f"Mean Total Run Time on the instances (s): {np.mean(runtime_sum)}\n")

        file.close()
