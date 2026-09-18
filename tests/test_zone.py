from app.analytics.zone import Zone


def test_point_inside_zone():

    zone = Zone(
        name="Test Zone",
        x1=100,
        y1=100,
        x2=500,
        y2=500
    )

    assert zone.contains(200, 200) is True


def test_point_outside_zone():

    zone = Zone(
        name="Test Zone",
        x1=100,
        y1=100,
        x2=500,
        y2=500
    )

    assert zone.contains(600, 600) is False


def test_point_on_zone_boundary():

    zone = Zone(
        name="Test Zone",
        x1=100,
        y1=100,
        x2=500,
        y2=500
    )

    assert zone.contains(100, 100) is True
    assert zone.contains(500, 500) is True