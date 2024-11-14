from Reading_dataset import *
from model_functions import *
from scipy.spatial.distance import euclidean
from Algorithm import RL_Agent
from VNS import *
from tqdm import tqdm
import time


## Cordaeu dataset coordinates are scatterred within a range of -100 to 100 => the max of X and Y is typically around 77-99
## So the coordinates should be first brought to a range of 0-200 and then, 
##   be scaled using division by 200 to be scaled to a range of 0-1 
## The maximum value for the depot coordinates is not close to 100, however, as we scaled the customers, we should follow the same path.
## Maximum Demand value in all dataset instances is 25; So, we can first scale them to 0-10 (as in the random datasets), 
##   and then, divide them by the capacity of the corresponding graph size. Alternatively, we can also scale them by 
##   normalizing the demand values (deviding all by the largest number which is 25) => Whichever gets better results is chosen
## Maximum service time is also 25 in all the instances
## Ready and due time horizons are around 200 to 700; So, we can divide them by 1000, as the due time for depots 
##   are from 0 to 1000, and the max horizon is considered to be 1.

if __name__ == "__main__":
    dataset = "Cordaeu"

    Total_Solution = dict()
    Total_Distances = dict()
    RunTime = dict()

    file_id = list(range(1, 21)) ## 1-21
    routes = {}
    training_times = {}
    
    for dataset_num_ind in tqdm(list(range(len(file_id))), desc=f"Instances: "):
        dataset_num = file_id[dataset_num_ind]
        print(f"============ {dataset}_{dataset_num} ============ ")
        if dataset_num <=9:
            data_path = "./Dataset/Public/cordeau2001-mdvrptw/pr0"+str(dataset_num)+".txt"
            ds_title = f"pr0{str(dataset_num)}"
        else:
            data_path = "./Dataset/Public/cordeau2001-mdvrptw/pr"+str(dataset_num)+".txt"
            ds_title = f"pr{str(dataset_num)}"

        routes[ds_title] = []
        data_df, depot_df, Vehicle_info, data_conf = reading_cordeu_ds(data_path)

        # data_df = data_df[0:20]
        # depot_df = depot_df[0:3]

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
        Max_VC = 1 #5

        depot_num = len(depot_df)
        vehicle_num = depot_num
        maxTime = Vehicle_info['max_T'][0]/10

        Max_VRT = maxTime
        n_episodes = 200
        n_vehicles = depot_num
        n_timesteps = max(data_df['DUE_DATE']) + 100

        alpha = 0.1
        beta = 0.2

        ########### Algorithms ##########
        distance_matrix = compute_distances(total_coords)
        # print("Distance Matrix Created")
        # print("================================================")

        start_time = time.time()

        # clusters = generating_clusters(C, D)
        # print("Clusters => ", clusters)

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
        
        # print("\n\nInitial routes >>", routes)

        initial_solution = dict()
        for depot in routes.keys():
            initial_solution[depot] = []
            for route in routes[depot].values():
                for r in route:
                    # initial_solution[depot].append([depot] + r + [depot])
                    initial_solution[depot].extend([depot] + r + [depot])

        print("\n\nInitial Solution >>", initial_solution)

        ############# VNS #############

        final_solution = dict()
        initial_distance = 0
        final_distance = 0
        for depot in initial_solution.keys():
            final_solution[depot] = []
            # for route in initial_solution[depot]:
            route = initial_solution[depot]
            init_dist = tour_distance(route, distance_matrix)
            initial_distance += init_dist
            # print("Initial Distance =>", init_dist)
            best_tour, exploration_time, exploitation_time = vns(route, depot, Demands, Max_VC, distance_matrix, TW_C, ServiceT_C, k_max=50, operator=two_opt)
            # print(best_tour, exploitation_time, exploration_time)
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

        Total_Solution[ds_title] = final_solution
        Total_Distances[ds_title] = {"Initial Solution": np.sum(initial_distance),
                                    "Sum": np.sum(final_distance),
                                    "Cust_Num": len(data_df),
                                    "Depot_Num": len(depot_df)}
        finish_time = time.time()
        RunTime[ds_title] = finish_time - start_time
        print(f"\n================ Evaluation For Instance {ds_title} Finished ======================\n")
    
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
