class DwellTimeTracker:

    def __init__(self):
        self.active_tracks = {}
        self.total_entries = 0
        self.completed_dwell_times = []

    def update(self, inside_ids, timestamp):
        inside_ids = set(inside_ids)

        entered_ids = []
        exited_ids = []

        # Detect new entries
        for track_id in inside_ids:

            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = timestamp
                self.total_entries += 1
                entered_ids.append(track_id)

        # Detect exits
        previous_ids = set(self.active_tracks.keys())

        for track_id in previous_ids - inside_ids:

            entered_at = self.active_tracks.pop(track_id)

            dwell_time = timestamp - entered_at

            self.completed_dwell_times.append(
                dwell_time
            )

            exited_ids.append(
                (track_id, dwell_time)
            )

        return entered_ids, exited_ids

    def get_current_dwell(self, track_id, timestamp):

        if track_id not in self.active_tracks:
            return 0.0

        return timestamp - self.active_tracks[track_id]

    def get_average_dwell(self):

        if not self.completed_dwell_times:
            return 0.0

        return (
            sum(self.completed_dwell_times)
            / len(self.completed_dwell_times)
        )