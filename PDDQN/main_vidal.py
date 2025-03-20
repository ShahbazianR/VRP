from Reading_dataset import *
from scipy.spatial.distance import euclidean
from Algorithm import RL_Agent
from tqdm import tqdm
import time
import copy

def tour_distance(tour, dist_matrix):
    return sum(dist_matrix[(tour[i-1],tour[i])] for i in range(len(tour)))

def compute_distances(clients):
    distances = dict()
    for from_client in clients.keys():
        for to_client in clients.keys():
            distances[(from_client, to_client)] = euclidean(clients[from_client], clients[to_client])
    return distances


if __name__ == "__main__":
    dataset = "Vidal"

    Total_Solution = dict()
    Total_Distances = dict()
    RunTime = dict()

    file_id = list(range(11, 25))
    dataset_subs = ['a', 'b']
    results = []
    training_times = {}
    for dataset_sub in dataset_subs:
        for dataset_num_ind in tqdm(list(range(len(file_id))), desc=f"Instances({dataset_sub}): "):
            dataset_num = file_id[dataset_num_ind]
            print(f"\n============ pr_{dataset_num}_{dataset_sub} ============ ")
            data_path = f"./Dataset/Public/vidal-al-2013-mdvrptw/pr"+str(dataset_num)+dataset_sub+".txt"
            ds_title = f"pr{dataset_num}{dataset_sub}"
            print("DS Title:", ds_title)

            data_df, depot_df, Vehicle_info, data_conf = reading_vidal_ds(data_path)

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
            Max_VC = 1

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
            agent, routes = RL_Agent(C, D, TW_C, Demands, Max_VC, ServiceT_C, n_vehicles, n_episodes, n_timesteps, \
                    Q_update_frequency = 2, Target_update_freq = 4, route_max_time = Max_VRT, 
                    alpha=alpha, beta=beta)

            print("\n\nRoutes >>", routes)

            solution = dict()
            for depot in routes.keys():
                solution[depot] = []
                for route in routes[depot].values():
                    for r in route:
                        solution[depot].append([depot] + r + [depot])

            distance = 0
            for depot in solution.keys():
                for route in solution[depot]:
                    init_dist = tour_distance(route, distance_matrix)
                    distance += init_dist

            print("Solution >>", solution)
            print("Distance >>", distance)
            print()


            Total_Solution[ds_title] = solution
            Total_Distances[ds_title] = {"Solution": solution,
                                         "Sum": np.sum(distance)}
            finish_time = time.time()
            RunTime[ds_title] = finish_time - start_time
            results.append({"DS":ds_title, "Solution": np.sum(distance), 
                            "Sum": np.sum(distance), "Time": RunTime[ds_title]})
            print(f"\n================ Evaluation For Instance {ds_title} Finished ======================\n")

            results_df = pd.DataFrame(results)
            results_df.to_csv('./PDDQN_Vidal.csv')

    file = open('Vidal_PDDQN_Log.txt', 'w')
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
        

