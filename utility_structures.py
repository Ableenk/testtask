class PositionsQueue():
    
    def __init__(self, queue_length):
        self.length = queue_length
        self.queue = [[] for _ in range(self.length)] 
        
    def add(self, active_id, amount, is_long):
        self.queue[-1].append((active_id, amount, is_long))
        
    def step(self):
        positions_to_handle = {}
        for queue_unit in self.queue[0]:
            active_id = queue_unit[0]
            amount = queue_unit[1]
            is_long = queue_unit[2]
            if active_id not in positions_to_handle:
                positions_to_handle[active_id] = []
            positions_to_handle[active_id].append((amount, is_long))
        for position in range(0, self.length-1):
            self.queue[position] = self.queue[position+1]
        self.queue[-1] = []
        return positions_to_handle