class OccupancyCounter:

    def __init__(self):
        self.current_ids = set()

    def update(self, tracked_ids):
        self.current_ids = set(tracked_ids)

    def count(self):
        return len(self.current_ids)