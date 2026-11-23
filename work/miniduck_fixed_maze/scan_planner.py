"""Small two-dimensional navigator driven by posed range scans and goal coordinates."""

from heapq import heappop, heappush
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


CELL = .045
BODY_MARGIN = .165
NEIGHBORS = (
    (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
)


class RangePlanner:
    def __init__(self, goal, bearings, max_range, padding=2., heuristic_weight=1.):
        self.goal = np.asarray(goal, dtype=float)
        self.bearings = np.asarray(bearings, dtype=float)
        self.max_range = max_range
        self.padding = np.asarray(padding, dtype=float)
        self.heuristic_weight = heuristic_weight
        self.origin = None
        self.free = None
        self.walls = None
        self.clearance = None
        self.cost = None
        self.path = np.empty((0, 2))
        self.target = None
        self.scan_count = 0

    def index(self, points):
        return np.rint((points - self.origin) / CELL).astype(int)

    def valid(self, cells):
        wx, wy = self.walls.shape
        return ((cells[..., 0] >= 0) & (cells[..., 0] < wx) &
                (cells[..., 1] >= 0) & (cells[..., 1] < wy))

    def update(self, position, yaw, hits):
        if self.origin is None:
            self.origin = np.minimum(position, self.goal) - self.padding
            shape = tuple((np.ceil((np.maximum(position, self.goal) + self.padding - self.origin) / CELL) + 1).astype(int))
            self.walls = np.zeros(shape, dtype=bool)
            self.free = np.zeros(shape, dtype=bool)
        if self.max_range <= 0:
            return
        self.scan_count += 1
        angles = self.bearings + yaw
        distances = np.full(len(angles), self.max_range)
        if len(hits):
            offsets = hits - position
            relative = np.arctan2(offsets[:, 1], offsets[:, 0]) - yaw
            bins = np.rint((relative - self.bearings[0]) * len(angles) / (2. * np.pi))
            distances[bins.astype(int) % len(angles)] = np.linalg.norm(offsets, axis=1)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        sample_radii = np.arange(0, self.max_range + CELL, .6 * CELL)
        xy = position + directions[:, None, :] * np.minimum(
            sample_radii[None, :], np.maximum(0., distances - CELL / 2.)[:, None]
        )[..., None]
        cells = self.index(xy.reshape((-1, 2)))
        safe = cells[self.valid(cells)]
        self.free[safe[:, 0], safe[:, 1]] = True
        if len(hits):
            blocks = self.index(hits)
            blocks = blocks[self.valid(blocks)]
            self.walls[blocks[:, 0], blocks[:, 1]] = True

    def replan(self, position):
        if self.scan_count == 0:
            return False
        self.clearance = CELL * distance_transform_edt(~self.walls)
        self.cost = 1. + .75 * (~self.free) + .09 / np.maximum(self.clearance, BODY_MARGIN)
        source = tuple(self.index(position))
        goal = tuple(self.index(self.goal))
        width, height = self.walls.shape
        blocked = self.clearance < BODY_MARGIN
        if not (self.valid(np.asarray(source)) and self.valid(np.asarray(goal))) or blocked[goal]:
            self.path = np.empty((0, 2))
            return False
        dist = {source: 0.}
        prev = {}
        queue = [(math.dist(source, goal), 0., source)]
        while queue:
            _, cost, node = heappop(queue)
            if cost != dist[node]:
                continue
            if node == goal:
                points = [node]
                while points[-1] != source:
                    points.append(prev[points[-1]])
                self.path = self.origin + CELL * np.asarray(points[::-1])
                return True
            x, y = node
            for dx, dy in NEIGHBORS:
                nxt = nx, ny = x + dx, y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height or blocked[nxt]:
                    continue
                if dx and dy and (blocked[x + dx, y] or blocked[x, y + dy]):
                    continue
                score = cost + (math.sqrt(2.) if dx and dy else 1.) * self.cost[nxt]
                if score < dist.get(nxt, math.inf):
                    dist[nxt] = score
                    prev[nxt] = node
                    heappush(queue, (score + self.heuristic_weight * math.dist(nxt, goal), score, nxt))
        self.path = np.empty((0, 2))
        return False

    def lookahead(self, position, distance=.36):
        if not len(self.path):
            return None
        near = int(np.argmin(np.linalg.norm(self.path - position, axis=1)))
        target = None
        for point in self.path[near + 1:]:
            gap = math.dist(point, position)
            if gap > distance:
                break
            checks = self.index(np.linspace(position, point, max(4, int(gap / (.5 * CELL)))))
            samples = self.clearance[checks[:, 0], checks[:, 1]]
            if np.any(samples[1:] < min(BODY_MARGIN, float(samples[0])) - .002):
                break
            target = point
        return target
