import numpy as np

from configurations import config
battery_capacity, vehicle_capacity, vehicle_velocity, \
    vehicle_energy_decay, energy_consumption_per_distance = config()

class Vehicle:
    vehicle_id = 0
    cx = 0
    cy = 0
    Max_cap = vehicle_capacity
    Max_battery = battery_capacity
    capacity = 0 ## Q regarding the total number of demands can be held by the vehicle
    current_charge = 0 ## 100 in percent
    departure_nodes = dict() ## Multi depots for both departure and arrival containing their ids and coordinates
    arrival_nodes = dict()
    current_departure_id = 0 ## the current departure node
    current_arrival_id = 0   ## the current arrival node
    max_travel_time = 1000
    current_travel_time = 0
    energy_decay = vehicle_energy_decay
    velocity = vehicle_velocity
    current_route_travel_time = 0
    trip_max_duration = 200

    def __init__(self):
        return

    def initiate(self, index, x, y, dep_nodes, arr_nodes, cap_max, max_T, energy_decay, battery_total_capacity, velocity):
        self.vehicle_id = index
        self.cx = x
        self.cy = y
        self.departure_nodes = dep_nodes
        self.arrival_nodes = arr_nodes
        self.capacity = cap_max
        self.max_cap = cap_max
        self.max_travel_time = max_T
        self.current_travel_time = 0
        self.current_charge = battery_total_capacity
        self.max_battery = battery_total_capacity
        self.energy_decay_per_distance = energy_decay
        if len(self.departure_nodes) > 1:
          depot_keys = list(self.departure_nodes.keys()) ## random selection of departure nodes
          departure_rand_index = depot_keys[np.random.randint(0,len(depot_keys)-1)]
          self.current_departure_id = departure_rand_index

          depot_keys = list(self.arrival_nodes.keys()) ## random selection of arrival nodes
          arrival_rand_index = depot_keys[np.random.randint(0,len(depot_keys)-1)]
          self.current_arrival_id = arrival_rand_index
        else:
          dep_key = list(self.departure_nodes.keys())[0]
          arr_key = list(self.arrival_nodes.keys())[0]
          self.current_departure_id = dep_key
          self.current_arrival_id = arr_key

    def set_current_depot_ids(self, departure, arrival, random = False):

        if len(self.departure_nodes) > 1:
          if random:
            depot_keys = list(self.departure_nodes.keys()) ## random selection of departure nodes
            rand_index = depot_keys[np.random.randint(0,len(depot_keys)-1)]
            self.current_departure_id = rand_index

            depot_keys = list(self.arrival_nodes.keys()) ## random selection of arrival nodes
            rand_index = depot_keys[np.random.randint(0,len(depot_keys)-1)]
            self.current_arrival_id = rand_index
          else:
            self.current_departure_id = departure
            self.current_arrival_id = arrival
        else: ## there is only one option
          dep_key = list(self.departure_nodes.keys())[0]
          arr_key = list(self.arrival_nodes.keys())[0]
          self.current_departure_id = dep_key
          self.current_arrival_id = arr_key


    def get_coordinates(self):
        return [self.cx, self.cy]

    def get_info(self):
          print("id ", self.vehicle_id,'\n',
                "cx ", self.cx, '\n',
                "cy ", self.cy,'\n',
                "Max_cap ", self.Max_cap,'\n',
                "Max_battery ", self.Max_battery,'\n',
                "capacity ", self.capacity,'\n',
                "current_charge ", self.current_charge,'\n',
                "departure_nodes ", self.departure_nodes,'\n',
                "arrival_nodes ", self.arrival_nodes,'\n',
                "current_departure_id ", self.current_departure_id,'\n',
                "current_arrival_id ", self.current_arrival_id,'\n',
                "max_travel_time ", self.max_travel_time,'\n',
                "current_travel_time ", self.current_travel_time,'\n',
                "energy_decay ", self.energy_decay
                )