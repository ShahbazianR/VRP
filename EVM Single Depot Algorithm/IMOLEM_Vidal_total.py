from classes import *
from functions import *
import pandas as pd
import re
from scipy.stats import pareto
from scipy.spatial.distance import euclidean
import time
from tqdm import tqdm
import re
import pandas as pd
def reading_vidal_ds(file_path):

    file = open(file_path, 'r')
    read_lines = file.readlines()
    type_number = re.split('\n', read_lines[0])[0]
    type_number = re.split(' ', type_number)

    vrp_type, n_vehicles, n_customers, days = type_number
    print(vrp_type, n_vehicles, n_customers)

    max_duration = []
    max_load = []
    read_lines = read_lines[1:]
    for i in range(int(days)):
        line = read_lines[i]
        line = re.split('\n', line)[0]
        MD, MQ = re.split(' ', line)
        max_duration.append(int(MD))
        max_load.append(int(MQ))

    depot_data = read_lines[-int(days):]
    data_lines = read_lines[int(days):-int(days)]
    data_df = []
    for line in data_lines:
        cust_line = re.split('\n', line)[0]
        cust_line = re.split('\t', line)
        cust_info = []
        for item in cust_line:
            if item != '':
                if '\n' in item:
                    item = re.split('\n', item)[0]
                cust_info.append(float(item))

        data_df.append(cust_info[0:5]+cust_info[-2:])
    
    data_df = pd.DataFrame(data_df, columns=['CUST_NO.', 'XCOORD.', 'YCOORD.', 'SERVICE_TIME', 'DEMAND', 'READY_TIME', 'DUE_DATE'])

    depot_df = []
    for line in depot_data:
        depot_line = re.split('\n', line)[0]
        depot_line = re.split('\t', line)
        depot_info = []
        for item in depot_line:
            if item != '':
                if '\n' in item:
                    item = re.split('\n', item)[0]
                depot_info.append(float(item))
        depot_df.append(depot_info[0:5]+depot_info[-2:])
    
    depot_df = pd.DataFrame(depot_df, columns=['CUST_NO.', 'XCOORD.', 'YCOORD.', 'SERVICE_TIME', 'DEMAND', 'READY_TIME', 'DUE_DATE'])


    vrp_types = {0:"VRP", 1:"PVRP", 2:"MDVRP", 3: "SDVRP", 4:"VRPTW", 5: "PVRPTW", 6: "MDVRPTW", 7:"SDVRPTW"}
    
    
    return data_df, depot_df, {'type':vrp_types[int(vrp_type)], 'n_vehicles':int(n_vehicles), 'n_customers':int(n_customers), 'days':int(days)}
    


if __name__ == "__main__":
    dataset = "Vidal"

    title = "pr"
    codes = ['a', 'b']
    file_id = list(range(11, 24))

    routes = {}
    times = {}

    for code in codes:
        for dataset_num in file_id:
            print(f"============ {dataset}_{title}{dataset_num}{code}=> Total ============ ")
            start_time = time.time()
            ## Data Preparation ##
            data_path = f"./Dataset/Public/vidal-al-2013-mdvrptw/pr{dataset_num}{code}.txt"
            file_name = f"{title}{dataset_num}{code}"

            print("File Path =>",data_path)
            print("File Name =>",file_name)

            print(f"\n=========== {ds_title} ===========\n")
            
            data_df, depot_df, data_conf = reading_vidal_ds(data_path)
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

            client_number = len(data_df)
            print(f"Client Numbers => {client_number}")

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

    ##        print(coordinates)
    ##        print(time_windows)
    ##        print(demands)
    ##        print(service_times)

            distance_matrix = compute_distances(list(coordinates.values()))
            print("Distance Matrix Shape=> ", distance_matrix.shape)

            
            #################################################################
            #################################################################
            iteration = 50
            #population_size = 100
            population_size = 2*client_number
            #n_generation = 200
            n_generation = 2*client_number
            length = len(coordinates)-1
            vehicle_capacity = 100
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
            print(f"P => {len(P)}")

            if not len(P):
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
                # print("Pareto sort", len(pt_P_values), pt_P_values)
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
            if len(results):
                print(f"Dataset {dataset}, {dataset_num}, {client_number}")
                x = float(np.mean(results, axis=0)[0])
                y = float(np.mean(results, axis=0)[1])
                z = float(np.mean(results, axis=0)[2])
                print("Mean Distance =>", x)
                print("Mean M Values =>", y)
                print("Mean N Subroutes =>", z)
                k = str(f"Dataset {dataset}_{dataset_num}_{code}, {client_number}")
                routes[k] = [x, y, z]
                finish_time = time.time()
                total_time = finish_time-start_time
                times[k] = total_time

    for item in routes:
        print(item, " => ", routes[item])
        print(item, " => ", times[item])
