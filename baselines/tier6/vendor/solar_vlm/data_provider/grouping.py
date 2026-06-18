# File: data_provider/grouping.py
"""Spatial plant-grouping for Solar-VLM's multi-station GNN.

Solar-VLM forecasts a *fixed-size* set of co-located stations jointly (GNN +
cross-station attention). Our dataset is individual PV plants on a disjoint
cross-plant split. To run Solar-VLM faithfully (GNN on) we cluster the plants of
ONE split partition into groups of exactly ``num_stations`` co-located plants by
latitude/longitude, so each group looks like a Solar-VLM "station set".

Grouping is done *within* a split partition (train groups use only train plants,
test groups only test plants) so the disjoint cross-plant contract is preserved
at the group level: a test group is a set of plants never seen in training.

Greedy nearest-neighbour chaining (deterministic): seed with the plant whose id
sorts first among the unused, then repeatedly attach the nearest unused plant
until the group has ``num_stations`` members. A trailing remainder smaller than
``num_stations`` is padded by repeating its nearest in-group members (documented
approximation — the GNN has no station mask, so padded duplicates only add
redundant, self-consistent nodes).
"""

from __future__ import annotations

import numpy as np


def _haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two lat/lon points (scalar or array)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_station_groups(
    coords: dict[str, tuple[float, float]],
    num_stations: int,
) -> list[list[str]]:
    """Cluster plants into co-located groups of exactly ``num_stations``.

    ``coords`` maps site_id -> (latitude, longitude). Returns a list of groups,
    each a list of ``num_stations`` site_ids (the last group padded by repetition
    if the plant count is not a multiple of ``num_stations``). Deterministic.
    """
    sites = sorted(coords)
    if not sites:
        return []
    lat = {s: float(coords[s][0]) for s in sites}
    lon = {s: float(coords[s][1]) for s in sites}

    unused = list(sites)
    groups: list[list[str]] = []
    while unused:
        seed = unused.pop(0)              # smallest remaining id (deterministic)
        group = [seed]
        while len(group) < num_stations and unused:
            d = _haversine(
                lat[seed], lon[seed],
                np.array([lat[s] for s in unused]),
                np.array([lon[s] for s in unused]),
            )
            j = int(np.argmin(d))
            group.append(unused.pop(j))
        if len(group) < num_stations:     # pad short trailing group by repetition
            k = 0
            while len(group) < num_stations:
                group.append(group[k % len(group)])
                k += 1
        groups.append(group)
    return groups
