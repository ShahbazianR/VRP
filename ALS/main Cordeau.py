from utils import *
from model_functions import *
from scipy.spatial.distance import euclidean
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

    Total_Solutions = dict()
    Total_Distances = dict()
    Total_Time_Costs = dict()
    RunTime = dict()
    not_seen = dict()

    file_id = list(range(1, 21)) ## 1-21
    routes = {}
    training_times = {}

    print("Cordeau Full DS")
    
    for dataset_num_ind in tqdm(list(range(len(file_id))), desc=f"Instances: "):
        dataset_num = file_id[dataset_num_ind]
        print(f"\n============ {dataset}_{dataset_num} ============ ")
        if dataset_num <=9:
            data_path = "./Dataset/Public/cordeau2001-mdvrptw/pr0"+str(dataset_num)+".txt"
            ds_title = f"pr0{str(dataset_num)}"
        else:
            data_path = "./Dataset/Public/cordeau2001-mdvrptw/pr"+str(dataset_num)+".txt"
            ds_title = f"pr{str(dataset_num)}"

        routes[ds_title] = []
        data_df, depot_df, Vehicle_info, data_conf = reading_cordeu_ds(data_path)

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
        total_time_windows = dict()

        index = 0
        for item in range(depot_num):
            coordinates_depots[index] = [depot_df["XCOORD."][item], list(depot_df["YCOORD."])[item]]
            time_windows_depots[index] = [depot_df["READY_TIME"][item], list(depot_df["DUE_DATE"])[item]]
            total_time_windows[index] = [depot_df["READY_TIME"][item], list(depot_df["DUE_DATE"])[item]]

            total_coords[index] = [depot_df["XCOORD."][item], list(depot_df["YCOORD."])[item]]
            index += 1

        for item in list(dict(data_df["XCOORD."]).keys()):
            coordinates_customers[index] = [data_df["XCOORD."][item], list(data_df["YCOORD."])[item]]
            time_windows_customers[index] = [data_df["READY_TIME"][item], list(data_df["DUE_DATE"])[item]]
            total_time_windows[index] = [data_df["READY_TIME"][item], list(data_df["DUE_DATE"])[item]]
            demands_customers[index] = list(data_df["DEMAND"])[item]
            service_times_customers[index] = list(data_df["SERVICE_TIME"])[item]

            total_coords[index] = [data_df["XCOORD."][item], list(data_df["YCOORD."])[item]]
            index += 1

        start_time = time.time()
        distance_matrix = compute_distances(total_coords)

        C = coordinates_customers
        D = coordinates_depots
        TWC = total_time_windows
        TTW = total_time_windows
        ST = service_times_customers
        Demands = demands_customers

        MAX_C = 1
        MAX_T = 20

        T_start = 400 
        beta = 0.99985  ## delta = beta^(iteration)*T_start

        N = 30000
        gamma = 5

        final_solution, RL = ALS(C, D, Demands, TTW, ST, MAX_C, MAX_T, distance_matrix, T_start, beta, N, gamma)

        final_routes = routes_decompose(final_solution, len(D.keys()))
        total_dist = 0
        total_time_eval = 0
        served = []
        seen_clients = []
        for route in final_routes:
            if len(route) > 2:
                dist = tour_distance(route, distance_matrix)
                total_dist += dist
                time_eval = route_time_eval(route, distance_matrix, TTW, ST)
                print(route, dist, total_dist, time_eval)
                total_time_eval += time_eval
                served.extend(route)
                seen_clients.extend(set(route))
        
        print("\nServed Clients")
        # print(sorted(served))
        print(len(served), len(set(served)))
        not_seen_count = 0
        for item in list(C.keys()):
            if item not in seen_clients:
                print("Not Seen =>", item)
                not_seen_count += 1
        
        print("Number of Not Seen Clients:", not_seen_count)

        print(f"Total Distance {dataset}{ds_title}: ", total_dist)
        finish_time = time.time()
        Total_Distances[ds_title] = total_dist
        Total_Solutions[ds_title] = final_routes
        Total_Time_Costs[ds_title] = total_time_eval
        RunTime[ds_title] = finish_time - start_time
        not_seen[ds_title] = {"Not_Seen":not_seen_count, "RL":RL}

    print("============================================")
    for key in Total_Distances.keys():
        print(f"{dataset}_{key} Distance:", Total_Distances[key])
        print(f"{dataset}_{key} Solution:", Total_Solutions[key])
        print(f"{dataset}_{key} Time Cost:", Total_Time_Costs[key])
        print(f"{dataset}_{key} Run Time:", RunTime[key])
        print(f"{dataset}_{key} Not_seen:", not_seen[key])
        print("=================================")
