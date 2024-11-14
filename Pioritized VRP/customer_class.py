class Customer:
    picked_up_flag = False
    id = 0
    demand = 0
    cx = 0
    cy = 0
    tw_start = 0
    tw_end = 0
    service_time = 0

    def __init__(self, index, cx, cy, start_tw, end_tw, quantity, service_time):
        self.id = index
        self.cx = cx
        self.cy = cy
        self.tw_start = start_tw
        self.tw_end = end_tw
        self.demand = quantity
        self.service_time = service_time

    def adapt_demand(self, new_demand):
        self.demand = new_demand

    def adapt_coordinates(self, x, y):
        self.cx = x
        self.cy = y

    def adapt_serive_time(self, new_service_time):
        self.service_time = new_service_time

    def adapt_time_window(self, s_tw, e_tw):
        self.tw_start = s_tw
        self.tw_end = e_tw

    def get_demand(self):
        return self.demand

    def get_coordinates(self):
        return [self.cx, self.cy]

    def get_service_time(self):
        return self.service_time

    def get_time_window(self):
        return [self.tw_start, self.tw_end]

    def picked_up(self):
        self.picked_up_flag = True

    def get_info(self):
      print("picked_up_flag ", self.picked_up_flag,'\n',
            "id ", self.id,'\n',
            "demand ", self.demand,'\n',
            "cx ", self.cx,'\n',
            "cy ", self.cy,'\n',
            "tw_start ", self.tw_start,'\n',
            "tw_end ", self.tw_end,'\n',
            "service_time ", self.service_time,'\n',
            )