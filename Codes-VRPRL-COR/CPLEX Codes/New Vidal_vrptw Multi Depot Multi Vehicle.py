import pandas as pd
import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import networkx as nx
import random
from tqdm import tqdm
import time
from utils import *
from datetime import date

rnd = np.random
rnd.seed(0)


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

    vehicles_dict = {'max_T': max_duration, 'max_load': max_load}

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
    
    
    return data_df, depot_df, vehicles_dict, {'type':vrp_types[int(vrp_type)], 'n_vehicles':int(n_vehicles), 'n_customers':int(n_customers), 'days':int(days)}
    


def vrptw_cplex(dataframe_client, data_depots, capacity=200, maxtime=500, vehicle_number=2, name="Fig_0"):
    print("Clients:\n", dataframe_client.head())
    print("Depots:\n", data_depots.head())

    depot_number=len(data_depots)

    depot_data = data_depots
    clients = dataframe_client

    ## Number of vehicles
    K = [i for i in range(vehicle_number)]

    M = 20
    n = len(clients)
    Q = capacity
    N = [i for i in range(len(depot_data), n + len(depot_data))] ## Clients
    D = [i for i in range(len(depot_data))] ## Depots
    V = D + N  ## All nodes
    
    demands = {}
    for i in V:
        if i in D:
            demands[i] = depot_data["DEMAND"][i]
        elif i in N:
            demands[i] = clients["DEMAND"][i-len(D)]

    # velocity = 20
    print("===============")
    print("demands", demands)
    print("V", V)
    print("D", D)
    print("N", N)
    print("K", K)
    print("===============\n")

    loc_x = [None]*len(V)
    loc_y = [None]*len(V)
    for i in range(len(V)):
        if i in D:
            loc_x[i] = depot_data["XCOORD."][i]
            loc_y[i] = depot_data["YCOORD."][i]
        elif i in N:
            loc_x[i] = clients["XCOORD."][i-len(D)]
            loc_y[i] = clients["YCOORD."][i-len(D)]

    print(loc_x)
    print(loc_y)
    print("===============\n")

    KV = [(i,j,k) for i in V for j in V for k in K if i!=j]
    KA = [(i, k) for i in V for k in K]
    KV2 = [(i, k) for i in V for k in K]

    # A: List of Arcs
    A = [(i, j) for i in V for j in V if i != j]  
    # c: Dictionary of distances/costs
    c = {(i, j): round(np.hypot(loc_x[i] - loc_x[j], loc_y[i] - loc_y[j])) for i, j in A}  
    # print(c)

    alpha=0.1
    beta=0.2

    T_max = max(clients["DUE_DATE"])
    T = max(clients["READY_TIME"])
    service_time = dict()
    tw_starts = dict()
    tw_ends = dict()
    for item in V:
        if i in D:
            tw_s = depot_data["READY_TIME"][i]
            tw_e = depot_data["DUE_DATE"][i]
            service_time[item] = depot_data["SERVICE_TIME"][i]
            tw_starts[item] = tw_s
            tw_ends[item] = tw_e

        elif i in N:
            index = i-len(D)
            tw_s = clients["READY_TIME"][index]
            tw_e = clients["DUE_DATE"][index]
            service_time[item] = clients["SERVICE_TIME"][index]
            tw_starts[item] = tw_s
            tw_ends[item] = tw_e

    print("===============")
    print(tw_starts)
    print(tw_ends)
    print("===============\n")

    # print("A => ", A)
    # print("K => ", K)

    # Create a CPLEX model:
    mdl = Model('MDVRPTW')
    # Define arcs and capacities:
    x = mdl.binary_var_dict(KV, name='x') ## Binary constraint (Each arc should be visited exactly once: visited and left)
    y = mdl.binary_var_dict(KA, name = 'y')
    z = mdl.binary_var_dict(KV, name='z')
    u = mdl.continuous_var_dict(KA, ub=Q, name='u') ## Vehicle capacity
    t = mdl.continuous_var_dict(KV2, ub=T, name='t') ## The arrival time
    a = mdl.integer_var_dict(tw_starts, name='a') ## Start time windows
    e = mdl.integer_var_dict(tw_ends, name='e') ## End time windows
    s = mdl.integer_var_dict(service_time, name='s') ## Service time

    b_tw = mdl.integer_var(0, name='btw')
    a_tw = mdl.integer_var(0, name='atw')


    # # Define objective function:
    mdl.minimize(mdl.sumsq(c[i, j] * x[i, j, k] for i, j, k in KV) + beta*b_tw + alpha*a_tw) ## Constraint 1 => Model's Objective

    # # Add constraints:
    mdl.add_constraints(mdl.sum(y[i, k] for k in K) == 1 for i in N) ## Constraint 3
    mdl.add_constraints(mdl.sum(x[i, j, k] for j in V if i!=j)==y[i, k] for i in N for k in K) ## Constraint 2

    mdl.add_constraints((mdl.sum(x[i, h, k] for i in V if i!=h)-mdl.sum(x[h, j, k] for j in V if h!=j))==0 for k in K for h in N) ## Constraint 4

##    mdl.add_constraints(mdl.sum(x[d, j, k] for d in D for j in N) <= 1 for k in K) ## Constraint 5
##    mdl.add_constraints(mdl.sum(x[i, d, k] for d in D for i in N) <= 1 for k in K) ## Constraint 6

    mdl.add_constraints(mdl.sum(x[d, j, k] for k in K for j in N) == 1 for d in D) ## Constraint 5
    mdl.add_constraints(mdl.sum(x[i, d, k] for k in K for i in N) == 1 for d in D) ## Constraint 6

    ################################ ## Starting and Ending in the Same Depot
    mdl.add_constraints(mdl.sum(x[d, c, k] for c in N)==mdl.sum(x[c, d, k] for c in N) for d in D for k in K)
    ################################


    mdl.add_indicator_constraints(
        mdl.indicator_constraint(x[i, j, k], u[j, k]==u[i,k] + demands[j]) for i in N for j in N for k in K if i!=j) ## Demand constraint

    mdl.add_constraints(u[i,k] >= demands[i] for i in N for k in K) ## Constraint 7
    
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(x[i, j, k], mdl.max(t[i, k] + c[i, j] + s[i] + M*(1-x[i,j, k]), a[j]) == t[j, k]) for i, j in A for k in K if not i in D and not j in D) ## Constraint 8
    

    #=================== Time Window Constraints ======================#
    # mdl.add_indicator_constraints(
    #     mdl.indicator_constraint(y[i, k], t[i, k]<=e[i]) for i, j in A for k in K if not i in D and not j in D) ## Constraint 9
    # mdl.add_indicator_constraints(
    #     mdl.indicator_constraint(y[i, k], t[i, k]>=a[i]) for i, j in A for k in K if not i in D and not j in D) ## Constraint 9
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(y[i,k], t[i,k]-e[i]<= alpha) for i,j in A for k in K if not i in D) ## Constraint 9: Soft Time Window Model_3
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(y[i,k], a[i]-t[i,k]<= beta) for i,j in A for k in K if not i in D) ## Constraint 9: Soft Time Window Model_3
    

    mdl.add_indicator_constraints(
        mdl.indicator_constraint(y[i,k], a_tw == a_tw + (a[i]-t[i,k])) for i,j in A for k in K if not i in D) ## Constraint 9: Soft Time Window Model_3
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(y[i,k], b_tw == b_tw + (t[i,k]-e[i])) for i,j in A for k in K if not i in D) ## Constraint 9: Soft Time Window Model_3
    #=================== Time Window Constraints ======================#


    mdl.add_indicator_constraints(
        mdl.indicator_constraint(x[i, j, k],  e[j]>=t[j, k]) for i, j in A for k in K if not i in D and not j in D)
    
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(x[i, j, k],  t[j, k]<=T_max) for i, j in A for k in K if not i in D and not j in D) ## Constriant 10

    mdl.add_constraints(t[i, k] == 0 for i in D for k in K) ## Constraint 11
    # mdl.add_indicator_constraints(
    #     mdl.indicator_constraint(x[i, j, k], t[i, k]==0) for i in D for j in N for k in K) ## Constraint 11
    
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(x[i, j, k], z[j, d, k]==1) for i in V for j in D for d in V for k in K if i!=j and i!=d and j!=d) ## Constraint 12
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(z[j, d, k], t[d, k]>=t[j, k]+M*(1-z[j,d, k])) for d in V for j in V for k in K if d!=j) ## Constraint 12


    mdl.parameters.timelimit = 1*15*60 #  2*60*60  Add running time limit

    # Solving model:
    solution = mdl.solve(log_output=True)

    if solution is not None:
        print(solution.solve_status)  # Returns if the solution is Optimal or just Feasible

        active_arcs = [a for a in KV if x[a].solution_value > 0.9]

        v_r = dict()
        for i in K:
            v_r[i] = []

        for route in active_arcs:
            for i in K:
                if route[2] == i:
                    v_r[i] += [(route[0], route[1])]

        # Plot solution:
        plt.figure()
        plt.scatter(loc_x[1:], loc_y[1:], c='b')
        for i in N:
            plt.annotate('$q_%d=%d$' % (i, demands[i]), (loc_x[i], loc_y[i]))

        colors = ['g', 'r', 'black', 'blue', 'y', 'orange', 'purple']
        c_id = 0
        for route in list(v_r.values()):
            for i, j in route:
                plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], c=colors[c_id], alpha=0.3)
            c_id += 1

        for i in D:
            plt.plot(loc_x[i], loc_y[i], c='r', marker='s')
            plt.annotate('$D_%d$' % (i), (loc_x[i] + 2, loc_y[i]))

        title = "Routes:\n"
        for item in v_r.keys():
            title += "V_"+str(item)+": "+str(v_r[item])+"\n"
        
        plt.text(10+min(clients["XCOORD."]), 20+min(clients["YCOORD."]), f'{title}', fontsize = 10) 
        plt.axis('equal')
        plt.title(f'Multi Depot cap:{capacity}; N: {len(V)}; D:{depot_number};', fontsize = 10)
        # plt.show()
        plt.savefig(f'./Figures/{name}.jpg', format='jpg')
        
        return v_r
    else:
        return None


if __name__ == "__main__":

    dataset = "Vidal"
    
    Total_Solution = dict()
    Total_Distances = dict()
    RunTime = dict()
    No_Solutions = list()

    length = 20
    depot_num = 3
    vehicle_num = 3

    codes = ['a', 'b']
    file_id = list(range(11, 24))
    title = "pr"
    for code in codes:
        for dataset_num_ind in tqdm(list(range(len(file_id))), desc=f"Instances: "):
            dataset_num = file_id[dataset_num_ind]
            print(f"============ {dataset}_{dataset_num}_{code} ============ ")
            data_path = f"./Dataset/Public/vidal-al-2013-mdvrptw/{title}{dataset_num}{code}.txt"
            ds_title = f"{title}{dataset_num}{code}"
                

            print("File Path =>",data_path)
            print("File Name =>",ds_title)


            data_df, depot_df, Vehicle_info, data_conf = reading_vidal_ds(data_path)

            data_df = data_df[0:length]
            depot_df = depot_df[0:depot_num]
            
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

            Max_VC = 5

            maxTime = Vehicle_info['max_T'][0]/10

            total_coords = dict()

            index = 0
            for item in range(depot_num):
                total_coords[index] = [depot_df["XCOORD."][item], list(depot_df["YCOORD."])[item]]
                index += 1

            for item in list(dict(data_df["XCOORD."]).keys()):
                total_coords[index] = [data_df["XCOORD."][item], list(data_df["YCOORD."])[item]]
                index += 1



            start_time = time.time()
            distance_matrix = compute_distances(total_coords)
            current_date = date.today()
            filename = f"Vidal_{ds_title} C_{length} D_{depot_num} V_{vehicle_num} DT_{str(current_date)}"

            routes = vrptw_cplex(data_df, depot_df, capacity=Max_VC, maxtime=maxTime, vehicle_number=vehicle_num, name=filename)
            finish_time = time.time()
            
            if routes is not None:
                route_distance = 0
                for depot in routes.keys():
                    for arc in routes[depot]:
                        arc_dist = distance_matrix[arc]
                        route_distance += arc_dist

                Total_Solution[ds_title] = routes
                Total_Distances[ds_title] = route_distance
                RunTime[ds_title] = finish_time - start_time

                print("Solution", routes)
                print("Distance", route_distance)
                print("Time", finish_time - start_time)

            else:
                No_Solutions.append(ds_title)
        

    distances_sum = []
    runtime_sum = []
    for key in Total_Solution.keys():
        print(key, "Solution: ", Total_Solution[key])
        print(key, "Distances per Depot: ", Total_Distances[key])
        distances_sum.append(Total_Distances[key])
        print("Run Time:" ,RunTime[key])
        runtime_sum.append(RunTime[key])
        print("==========================\n")

    print("Mean Total Sum Distances on the instances: ", np.mean(distances_sum))
    print("Mean Total Run Time on the instances (s): ", np.mean(runtime_sum))
    print("No Solution Cases: ", No_Solutions)
