from app.analytics.dwell_time import DwellTimeTracker


def test_new_person_enters_zone():

    tracker = DwellTimeTracker()

    entered, exited = tracker.update(
        {1},
        10.0
    )

    assert entered == [1]
    assert exited == []
    assert tracker.total_entries == 1


def test_person_exits_zone_and_dwell_is_recorded():

    tracker = DwellTimeTracker()

    tracker.update(
        {1},
        10.0
    )

    entered, exited = tracker.update(
        set(),
        15.0
    )

    assert entered == []
    assert exited == [(1, 5.0)]


def test_current_dwell_time():

    tracker = DwellTimeTracker()

    tracker.update(
        {1},
        10.0
    )

    dwell = tracker.get_current_dwell(
        1,
        13.5
    )

    assert dwell == 3.5


def test_average_dwell_time():

    tracker = DwellTimeTracker()

    tracker.update({1}, 10.0)
    tracker.update(set(), 15.0)

    tracker.update({2}, 20.0)
    tracker.update(set(), 30.0)

    assert tracker.get_average_dwell() == 7.5