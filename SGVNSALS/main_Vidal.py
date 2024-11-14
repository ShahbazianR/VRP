from functions import *
from model_functions import *
from scipy.spatial.distance import euclidean
import time

if __name__ == "__main__":

    dataset_range = list(range(11, 25))

    routes = dict()
    distances = dict()
    vehicle_count = dict()
    RunTime = dict()

    codes = ['a', 'b']
    title = "pr"

    for code in codes:
      for dataset_num in dataset_range:
        ## Data Preparation ##
        data_path = f"./Dataset/Public/vidal-al-2013-mdvrptw/{title}{dataset_num}{code}.txt"
        ds_title = f"{title}{dataset_num}{code}"

        print(f"\n============== {ds_title} ==============")

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
        # print("Time Windows =>\n",time_windows_depots)

        # print("\n ======== Customers ========")
        # print("Coordinates =>\n", coordinates_customers, '\n')
        # print("Time Windows =>\n",time_windows_customers, '\n')
        # print("Demands =>\n",demands_customers, '\n')
        # print("Service Times =>\n",service_times_customers)

        start_time = time.time()
        distance_matrix = compute_distances(list(coordinates_customers.values()))
        print("Distance Matrix Shape=> ", distance_matrix.shape)

        print("\n======== SGVNSALS Algorithm ========\n")

        C = coordinates_customers
        D = coordinates_depots
        TW_C = time_windows_customers
        ServiceT_C = service_times_customers
        Demands = demands_customers

        lamda = 0.6 ## From Empirical Tests
        iterMax = 500
        maxTime = 20 #Vehicle_info['max_T'][0]
        Vehicle_info = {'max_T': [maxTime], 'max_load': [1]}
        maxLevel = 5
        pLS1 = 0.3
        pLS2 = 0.3
        als = True

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
        for item in list(C.keys()):
            if item not in seen_clients:
                print("Not Seen =>", item)

        print("\nCalculating the Distances in the Solution:")
        Export_Solution = {}
        evaluation_distances = []
        routes_num = 0
        for key in final_solution.keys():
            Export_Solution[key] = []
            dist = 0
            r = final_solution[key]
            print(f"Depot: {key}, Total Associated Routes: {r}")
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
                print(f"Depot: {key}, Route: {route}, Distance so far: {dist}")
            print(f"Distance for All the Routes Associated with the Depot {key}: ", dist)
            evaluation_distances.append(dist)

        print("Number of vehicles =>", routes_num)
        print(evaluation_distances)
        print(f"Mean distance in evaluation {len(data_df)} data {np.mean(evaluation_distances)}")
        print(f"Sum distance in evaluation {len(data_df)} data {np.sum(evaluation_distances)}")
        print(f"Extracted Solution:\n {Export_Solution}")

        routes[ds_title] = Export_Solution
        distances[ds_title] = {"Mean": np.mean(evaluation_distances), "Sum": np.sum(evaluation_distances)}
        vehicle_count[ds_title] = routes_num
        finish_time = time.time()
        RunTime[ds_title] = finish_time - start_time
        print(f"\n================ Evaluation For Client Number {len(data_df)} Finished ======================\n")

    for key in routes.keys():
        print(key, "Solution: ", routes[key])
        print(key, "Distances per Depot: ", distances[key])
        print(key, "Vehicle Number: ", vehicle_count[key])
        print("==========================\n")
