"""Timepix3-style pixel clustering.

This module implements the online clustering logic described in
"Real-time Timepix3 data clustering, visualization and classification with a
new Clusterer framework" (Meduna et al., CTD/WIT 2019).

The expected input is one hit per row with at least x, y and time columns.  A
pixel joins an event when it is 8-connected to the event and the whole event
remains inside the configured time window.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Iterable, Mapping


NEIGHBOR_OFFSETS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True, slots=True)
class PixelEvent:
    """A finalized cluster/event of Timepix3 pixels."""

    id: int
    pixels: tuple[dict[str, Any], ...]
    positions: tuple[int, ...]
    indices: tuple[Any, ...]
    min_time_ns: float
    max_time_ns: float
    coordinates: frozenset[tuple[int, int]]

    @property
    def size(self) -> int:
        return len(self.pixels)

    @property
    def duration_ns(self) -> float:
        return self.max_time_ns - self.min_time_ns


@dataclass(frozen=True, slots=True)
class _TimePixel:
    position: int
    index: Any
    x: int
    y: int
    time_ns: float
    record: dict[str, Any]


class _OpenCluster:
    __slots__ = (
        "pixels",
        "coords",
        "accepted_coords",
        "min_time_ns",
        "max_time_ns",
        "_version",
    )

    def __init__(self, pixel: _TimePixel) -> None:
        self.pixels: list[_TimePixel] = []
        self.coords: set[tuple[int, int]] = set()
        self.accepted_coords: set[tuple[int, int]] = set()
        self.min_time_ns = pixel.time_ns
        self.max_time_ns = pixel.time_ns
        self._version = 0
        self.add(pixel)

    def __len__(self) -> int:
        return len(self.pixels)

    def can_add(self, pixel: _TimePixel, time_window_ns: float) -> bool:
        if not _time_span_fits(
            self.min_time_ns,
            self.max_time_ns,
            pixel.time_ns,
            pixel.time_ns,
            time_window_ns,
        ):
            return False

        return (pixel.x, pixel.y) in self.accepted_coords

    def add(self, pixel: _TimePixel) -> None:
        self.pixels.append(pixel)
        self.min_time_ns = min(self.min_time_ns, pixel.time_ns)
        self.max_time_ns = max(self.max_time_ns, pixel.time_ns)

        coord = (pixel.x, pixel.y)
        self.coords.add(coord)
        self.accepted_coords.add(coord)
        for dx, dy in NEIGHBOR_OFFSETS_8:
            self.accepted_coords.add((pixel.x + dx, pixel.y + dy))

    def merge(self, other: "_OpenCluster") -> None:
        # Move pixels one by one so the coordinate and neighbor indexes stay in
        # sync with the merged event.
        for pixel in other.pixels:
            self.add(pixel)

    def to_event(self, event_id: int) -> PixelEvent:
        pixels = tuple(pixel.record for pixel in self.pixels)
        positions = tuple(pixel.position for pixel in self.pixels)
        indices = tuple(pixel.index for pixel in self.pixels)
        return PixelEvent(
            id=event_id,
            pixels=pixels,
            positions=positions,
            indices=indices,
            min_time_ns=self.min_time_ns,
            max_time_ns=self.max_time_ns,
            coordinates=frozenset(self.coords),
        )


class _ActiveClusters:
    def __init__(self) -> None:
        self.clusters: set[_OpenCluster] = set()
        self.acceptance_index: defaultdict[tuple[int, int], set[_OpenCluster]]
        self.acceptance_index = defaultdict(set)
        self.close_heap: list[tuple[float, int, int, _OpenCluster]] = []
        self._sequence = 0

    def add(self, cluster: _OpenCluster) -> None:
        self.clusters.add(cluster)
        self._register(cluster)
        self._push_close_entry(cluster)

    def remove(self, cluster: _OpenCluster) -> None:
        if cluster not in self.clusters:
            return
        self.clusters.remove(cluster)
        self._unregister(cluster)
        cluster._version += 1

    def candidates_for(self, pixel: _TimePixel, time_window_ns: float) -> list[_OpenCluster]:
        coord = (pixel.x, pixel.y)
        clusters = self.acceptance_index.get(coord)
        if not clusters:
            return []
        return [
            cluster
            for cluster in clusters
            if cluster in self.clusters and cluster.can_add(pixel, time_window_ns)
        ]

    def close_older_than(self, threshold_ns: float) -> list[_OpenCluster]:
        closed: list[_OpenCluster] = []
        while self.close_heap and self.close_heap[0][0] < threshold_ns:
            _, version, _, cluster = heappop(self.close_heap)
            if cluster not in self.clusters or cluster._version != version:
                continue
            self.remove(cluster)
            closed.append(cluster)
        return closed

    def remaining(self) -> list[_OpenCluster]:
        clusters = list(self.clusters)
        for cluster in clusters:
            self.remove(cluster)
        return clusters

    def _register(self, cluster: _OpenCluster) -> None:
        cluster._version += 1
        for coord in cluster.accepted_coords:
            self.acceptance_index[coord].add(cluster)

    def _unregister(self, cluster: _OpenCluster) -> None:
        for coord in cluster.accepted_coords:
            clusters = self.acceptance_index.get(coord)
            if clusters is None:
                continue
            clusters.discard(cluster)
            if not clusters:
                del self.acceptance_index[coord]

    def _push_close_entry(self, cluster: _OpenCluster) -> None:
        self._sequence += 1
        heappush(
            self.close_heap,
            (cluster.max_time_ns, cluster._version, self._sequence, cluster),
        )


def cluster_dataframe(
    df: Any,
    *,
    x_col: str = "x",
    y_col: str = "y",
    time_col: str = "time",
    time_window_ns: float = 2_000.0,
    unordered_window_ns: float = 200_000.0,
    cluster_col: str = "cluster_id",
    sort_by_time: bool = True,
    return_events: bool = True,
) -> Any:
    """Cluster a pandas-like dataframe of Timepix3 pixels.

    Returns a copy of ``df`` with a cluster id column.  By default it returns
    ``(clustered_df, events)``; set ``return_events=False`` to return only the
    dataframe.
    """

    events = cluster_pixels(
        df,
        x_col=x_col,
        y_col=y_col,
        time_col=time_col,
        time_window_ns=time_window_ns,
        unordered_window_ns=unordered_window_ns,
        sort_by_time=sort_by_time,
    )

    cluster_ids: list[int | None] = [None] * len(df)
    for event in events:
        for position in event.positions:
            cluster_ids[position] = event.id

    clustered = df.copy()
    clustered[cluster_col] = cluster_ids
    if return_events:
        return clustered, events
    return clustered


def cluster_pixels(
    pixels: Any,
    *,
    x_col: str = "x",
    y_col: str = "y",
    time_col: str = "time",
    time_window_ns: float = 2_000.0,
    unordered_window_ns: float = 200_000.0,
    sort_by_time: bool = True,
    return_labels: bool = False,
) -> list[PixelEvent] | tuple[list[PixelEvent], dict[int, int]]:
    """Cluster an iterable of pixel hits into Timepix3-style events.

    Parameters
    ----------
    pixels:
        A pandas dataframe, an iterable of mappings, an iterable of objects with
        x/y/time attributes, or an iterable of ``(x, y, time)`` sequences.
    time_window_ns:
        Maximum allowed time span inside one event.  The paper uses a
        conservative value of 2000 ns.
    unordered_window_ns:
        How long to keep open clusters before finalizing them.  Timepix3 streams
        can be partially unordered over about 200 microseconds.
    sort_by_time:
        Sort all pixels by time before clustering.  This is usually best for an
        offline dataframe.  Use ``False`` to mimic the original stream order.
    return_labels:
        Also return a mapping from input position to event id.
    """

    if time_window_ns <= 0:
        raise ValueError("time_window_ns must be positive")
    if unordered_window_ns < 0:
        raise ValueError("unordered_window_ns must be non-negative")

    timepixels = list(_iter_timepixels(pixels, x_col, y_col, time_col))
    if sort_by_time:
        timepixels.sort(key=lambda pixel: pixel.time_ns)

    active = _ActiveClusters()
    closed_clusters: list[_OpenCluster] = []

    for pixel in timepixels:
        candidates = active.candidates_for(pixel, time_window_ns)

        if not candidates:
            active.add(_OpenCluster(pixel))
        else:
            group = _compatible_merge_group(pixel, candidates, time_window_ns)
            primary = max(group, key=len)
            active.remove(primary)
            primary.add(pixel)

            # Join clusters connected by the new pixel, like the paper's
            # ProcessPixel algorithm.  Larger primary clusters reduce copying.
            for cluster in group:
                if cluster is primary:
                    continue
                active.remove(cluster)
                primary.merge(cluster)
            active.add(primary)

        threshold = pixel.time_ns - unordered_window_ns
        closed_clusters.extend(active.close_older_than(threshold))

    closed_clusters.extend(active.remaining())
    events = _finalize_events(closed_clusters)
    labels = {
        position: event.id
        for event in events
        for position in event.positions
    }

    if return_labels:
        return events, labels
    return events


def _compatible_merge_group(
    pixel: _TimePixel,
    candidates: list[_OpenCluster],
    time_window_ns: float,
) -> list[_OpenCluster]:
    # If a pixel can touch several clusters, only merge the subset whose
    # combined time span still satisfies the event definition.
    group: list[_OpenCluster] = []
    min_time = pixel.time_ns
    max_time = pixel.time_ns

    for cluster in sorted(candidates, key=len, reverse=True):
        next_min = min(min_time, cluster.min_time_ns)
        next_max = max(max_time, cluster.max_time_ns)
        if next_max - next_min < time_window_ns:
            group.append(cluster)
            min_time = next_min
            max_time = next_max

    if not group:
        # Each candidate was individually compatible, so this branch is only a
        # defensive fallback for unusual floating point inputs.
        return [candidates[0]]
    return group


def _finalize_events(clusters: list[_OpenCluster]) -> list[PixelEvent]:
    clusters.sort(key=lambda cluster: (cluster.min_time_ns, cluster.max_time_ns))
    return [cluster.to_event(event_id) for event_id, cluster in enumerate(clusters)]


def _time_span_fits(
    min_time_a: float,
    max_time_a: float,
    min_time_b: float,
    max_time_b: float,
    time_window_ns: float,
) -> bool:
    return max(max_time_a, max_time_b) - min(min_time_a, min_time_b) < time_window_ns


def _iter_timepixels(
    pixels: Any,
    x_col: str,
    y_col: str,
    time_col: str,
) -> Iterable[_TimePixel]:
    if _has_dataframe_columns(pixels, x_col, y_col, time_col):
        x_values = pixels[x_col].to_numpy(copy=False)
        y_values = pixels[y_col].to_numpy(copy=False)
        time_values = pixels[time_col].to_numpy(copy=False)
        index_values = pixels.index.to_numpy(copy=False)

        for position, (index, x, y, time_ns) in enumerate(
            zip(index_values, x_values, y_values, time_values)
        ):
            x_int = int(x)
            y_int = int(y)
            time_float = float(time_ns)
            yield _TimePixel(
                position=position,
                index=index,
                x=x_int,
                y=y_int,
                time_ns=time_float,
                record={x_col: x_int, y_col: y_int, time_col: time_float},
            )
        return

    if hasattr(pixels, "iterrows"):
        for position, (index, row) in enumerate(pixels.iterrows()):
            yield _make_timepixel(row, position, index, x_col, y_col, time_col)
        return

    for position, row in enumerate(pixels):
        yield _make_timepixel(row, position, position, x_col, y_col, time_col)


def _has_dataframe_columns(
    pixels: Any,
    x_col: str,
    y_col: str,
    time_col: str,
) -> bool:
    columns = getattr(pixels, "columns", None)
    if columns is None or not hasattr(pixels, "index"):
        return False
    return x_col in columns and y_col in columns and time_col in columns


def _make_timepixel(
    row: Any,
    position: int,
    index: Any,
    x_col: str,
    y_col: str,
    time_col: str,
) -> _TimePixel:
    x = int(_value(row, x_col, 0))
    y = int(_value(row, y_col, 1))
    time_ns = float(_value(row, time_col, 2))

    if isinstance(row, Mapping):
        record = dict(row)
    elif hasattr(row, "to_dict"):
        record = row.to_dict()
    else:
        record = {x_col: x, y_col: y, time_col: time_ns}

    return _TimePixel(
        position=position,
        index=index,
        x=x,
        y=y,
        time_ns=time_ns,
        record=record,
    )


def _value(row: Any, name: str, sequence_index: int) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    if hasattr(row, "__getitem__"):
        try:
            return row[name]
        except (KeyError, IndexError, TypeError):
            return row[sequence_index]
    return getattr(row, name)
