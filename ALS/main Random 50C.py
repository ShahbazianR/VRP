from utils import *
from model_functions import *
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import time


if __name__ == "__main__":
    Total_Solutions = dict()
    Total_Distances = dict()
    Total_Time_Costs = dict()
    RunTime = dict()
    request_lists = dict()

    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_20_2024-04-07.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_50_2024-04-07.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_VRP_100_2024-04-07.pkl'

    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_4D_VRP_20_2024-04-14.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_4D_VRP_50_2024-04-14.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_4D_VRP_100_2024-04-14.pkl'

    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_5D_VRP_20_2024-04-14.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_5D_VRP_50_2024-04-14.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_5D_VRP_100_2024-04-14.pkl'

    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_2D_VRP_20_2024-04-14.pkl'
    val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_2D_VRP_50_2024-04-14.pkl'
    # val_set_path = './Dataset/Random/Uniform Distribution/UValidation_dataset_2D_VRP_100_2024-04-14.pkl'  

    
    validation_dataset = read_from_pickle(val_set_path)
    print("Validation Dataset", val_set_path)
    n_depots = 2

    instances = []
    for x in validation_dataset.batch(1):
        depots, customers, demand, time_windows, service_times = x
        cust_vectors, depot_vectors = create_vectors(customers, depots, demand, time_windows)
        cust_vectors = tf.convert_to_tensor(cust_vectors, dtype=tf.float64)
        depot_vectors = tf.convert_to_tensor(depot_vectors, dtype=tf.float64)
        instances.append((cust_vectors, demand, time_windows, service_times, depot_vectors, customers, depots))
    
    for instance_id in tqdm(list(range(len(instances))), desc="Instances: "):
        print(f"\n============= Instance {instance_id} =============")
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

        N = 5000
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
        print(sorted(served))
        print(len(served), len(set(served)))
        not_seen_count = 0
        for item in list(C.keys()):
            if item not in seen_clients:
                print("Not Seen =>", item)
                not_seen_count += 1
        
        print("Number of Not Seen Clients:", not_seen_count)

        print(f"Total Distance {instance_id}: ", total_dist)
        finish_time = time.time()
        Total_Distances[instance_id] = total_dist
        Total_Solutions[instance_id] = final_routes
        Total_Time_Costs[instance_id] = total_time_eval
        RunTime[instance_id] = finish_time - start_time
        request_lists[instance_id] = {"Not_Seen":not_seen_count, "RL":RL}
        print(f"\n================ Evaluation For Instance {instance_id} Finished ======================\n")


    distances_sum = []
    runtime_sum = []
    time_cost = []
    print("============================================")
    for key in Total_Distances.keys():
        distances_sum.append(Total_Distances[key])
        runtime_sum.append(RunTime[key])
        time_cost.append(Total_Time_Costs[key])

        print(f"{key} Distance:", Total_Distances[key])
        print(f"{key} Solution:", Total_Solutions[key])
        print(f"{key} Time Cost:", Total_Time_Costs[key])
        print(f"{key} Run Time:", RunTime[key])
        print(f"{key} RL:", request_lists[key])
        print("=================================")
    
    print("Mean Total Sum Distances on the instances: ", np.mean(distances_sum))
    print("Mean Total Run Time on the instances (s): ", np.mean(runtime_sum))
    print("Mean Total Time Cost on the instances (s): ", np.mean(time_cost))
