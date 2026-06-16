# src/routing/speed_converter.py
# This module implements the conversion from predicted traffic flow to travel time.

import math


def flow_to_speed(flow: float) -> float:
    """
    Solve the quadratic equation: flow = -1.4648375 * speed^2 + 93.75 * speed
    Rewritten: 1.4648375 * speed^2 - 93.75 * speed + flow = 0

    Check for speed limit capping:
    - If predicted flow is at or below 351 vehicles per hour, the speed is capped at 60 km/h.
    - Otherwise, use the calculated speed.
    """
    if flow <= 351.0:
        return 60.0

    a = 1.4648375
    b = -93.75
    c = flow

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        # If flow is above capacity (discriminant < 0), speed drops to the capacity speed (32 km/h)
        return 32.0

    # Use the positive root since the road is assumed to be under capacity (Green curve)
    speed = (-b + math.sqrt(discriminant)) / (2 * a)
    return speed


def calculate_travel_time(
    speed: float, distance: float, is_intersection: bool = False
) -> float:
    """
    Calculate travel time in seconds.

    :param speed: speed in km/h
    :param distance: distance in km
    :param is_intersection: add 30s delay if true
    :return: travel time in seconds
    """
    if speed <= 0:
        speed = 32.0  # Fallback to avoid division by zero

    travel_time_seconds = (distance / speed) * 3600.0

    if is_intersection:
        travel_time_seconds += 30.0

    return travel_time_seconds
