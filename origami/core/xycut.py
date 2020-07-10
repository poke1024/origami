# quite similar to the ideas described in:

# Ha, Jaekyu & Haralick, Robert & Phillips, Ihsin. (1995).
# Recursive X-Y cut using bounding boxes of connected components.
# Proceedings of the Third International Conference on Document Analysis
# and Recognition. 2. 952-955. 10.1109/ICDAR.1995.602059


import logging
import collections
import numpy as np

from itertools import chain
from functools import partial

from origami.core.math import inset_bounds


Candidate = collections.namedtuple(
	'Candidate', ['axis', 'x', 'score', 'overlap'])


class Partition(collections.namedtuple(
	'Partition', ['a', 'b', 'overlap'])):

	def __iter__(self):
		return iter([self.a, self.b])


class Gap(collections.namedtuple(
	'Gap', ['axis', 'minu', 'minv', 'maxu', 'maxv'])):

	@property
	def u(self):
		return self.minu, self.maxu

	@property
	def v(self):
		return self.minv, self.maxv

	@property
	def du(self):
		return self.maxu - self.minu

	@property
	def dv(self):
		return self.maxv - self.minv

	@property
	def x(self):
		return [self.u, self.v][self.axis]

	@property
	def y(self):
		return [self.u, self.v][1 - self.axis]
	
	@property
	def bounds(self):
		minx, maxx = self.x
		miny, maxy = self.y
		return minx, miny, maxx, maxy


def _offset(x0, x1, amount):
	mid = (x0 + x1) / 2
	return min(x0 + amount, mid), max(x1 - amount, mid)


class Box:
	def __init__(self, name, minx, miny, maxx, maxy):
		self._name = name

		self._p = np.array([
			[minx, miny],
			[maxx, maxy]])

	def offset(self, amount):
		minx, maxx = _offset(*self._p[:, 0], amount)
		miny, maxy = _offset(*self._p[:, 1], amount)
		return Box(self._name, minx, miny, maxx, maxy)

	@property
	def name(self):
		return self._name

	@property
	def coords(self):
		return self._p

	@property
	def bounds(self):
		return self._p.flatten()


class Coordinates:
	def __init__(self, objs, axis, min_extent=0.1):
		self._objs = objs
		self._axis = axis

		xs = np.array([coords[:, axis] for coords in objs])
		ys = np.array([coords[:, 1 - axis] for coords in objs])

		xs[xs[:, 0] == xs[:, 1], 1] += min_extent
		ys[ys[:, 0] == ys[:, 1], 1] += min_extent

		self._min_by_label = np.min(xs, axis=-1)
		self._max_by_label = np.max(xs, axis=-1)
		self._ext_by_label = np.max(ys, axis=-1) - np.min(ys, axis=-1)

		self._ext_min = np.min(ys)
		self._ext_max = np.max(ys)

		c = np.hstack(xs)
		i = np.repeat(range(len(objs)), 2)

		s = np.argsort(c)

		self._x = c[s]
		self._label = i[s]

	def split_at(self, c):
		mask = self._x <= c

		# fix overlaps.
		a = set(self._label[mask])
		b = set(self._label[np.logical_not(mask)])

		for i in a & b:
			if abs(c - self._min_by_label[i]) < abs(c - self._max_by_label[i]):
				a.remove(i)
			else:
				b.remove(i)

		a = list(a)
		b = list(b)

		if not a:
			k = np.argmin([self._min_by_label[i] for i in b])
			a.append(b[k])
			del b[k]
		elif not b:
			k = np.argmax([self._max_by_label[i] for i in a])
			b.append(a[k])
			del a[k]

		return a, b

	def items(self):
		return zip(self._x, self._label)

	def candidate_splits(self, score, eps):
		active_set = collections.defaultdict(int)
		items = list(self.items())

		for (x0, i0), (x1, i1) in zip(items, items[1:]):
			active_set[i0] += 1
			if active_set[i0] == 2:
				del active_set[i0]

			if x0 > self._x[0] + eps:
				n = len(active_set)
				if n == 0:  # no overlaps.
					gap = Gap(
						axis=self._axis,
						minu=x0,
						minv=self._ext_min,
						maxu=x1,
						maxv=self._ext_max)
					yield Candidate(
						self, x0, score(gap), overlap=False)
				elif n >= 1:  # overlaps.
					err = 0
					for j in active_set.keys():
						err += self._ext_by_label[j] * min(
							abs(x0 - self._min_by_label[j]),
							abs(x0 - self._max_by_label[j]))
					yield Candidate(
						self, x0, -err, overlap=True)


default_scores = dict(
	largest_area=lambda gap: gap.du * gap.dv,
	widest_gap=lambda gap: gap.du,
	longest_cut=lambda gap: gap.dv)


class XYCut:
	def __init__(self, objs, score="widest_gap", eps=0, min_extent=0.1):
		if isinstance(score, str):
			score = default_scores[score]

		if len(objs) >= 2:
			coords = [np.array(o.coords, dtype=np.float64) for o in objs]
			lcs = [Coordinates(coords, axis, min_extent=min_extent) for axis in (0, 1)]
			splits = list(chain(*[lc.candidate_splits(score=score, eps=eps) for lc in lcs]))
			self._coords = np.array(coords)
		else:
			splits = None
			self._coords = None

		if not splits:
			self._split = None
			self._axis = None
			self._x = None
		else:
			best = max(splits, key=lambda x: x.score)

			ia, ib = best.axis.split_at(best.x)
			self._split = [objs[i] for i in ia], [objs[i] for i in ib]

			self._axis = lcs.index(best.axis)
			self._x = best.x
			self._overlap = best.overlap

	@property
	def valid(self):
		return self._split is not None

	def __iter__(self):
		return iter(self._split)

	def __getitem__(self, i):
		return self._split[i]

	@property
	def axis(self):
		return self._axis

	@property
	def x(self):
		return self._x

	@property
	def extent(self):
		a = 1 - self.axis
		coords = self._coords[:, :, a]
		return np.min(coords), np.max(coords)

	@property
	def overlap(self):
		return self._overlap


def _rxy_cut(boxes, **kwargs):
	if len(boxes) <= 1:
		return Partition([*boxes], [], False)

	cut = XYCut(boxes, **kwargs)
	if not cut.valid:
		return Partition([*boxes], [], len(boxes) > 1)

	if max(len(cut[0]), len(cut[1])) < len(boxes):
		a, b = map(partial(_rxy_cut, **kwargs), cut)
		return Partition(a, b, cut.overlap)
	else:
		logging.info("aborting _rxy_cut (%d elements)." % len(boxes))
		return Partition([*boxes], [], cut.overlap)


def _flatten(partition, leafs, rename):
	if isinstance(partition, Box):
		leafs.append(rename(partition))
	else:
		for x in partition:
			_flatten(x, leafs, rename)


def _groups(partition, groups, rename):
	if isinstance(partition, list) or partition.overlap:
		leafs = []
		_flatten(partition, leafs, rename)
		if leafs:
			groups.append(leafs)
	else:
		for x in partition:
			_groups(x, groups, rename)


_modes = dict(flat=_flatten, grouped=_groups)


def _reading_order(boxes, mode="flat", **kwargs):
	results = []
	_modes[mode](
		_rxy_cut(boxes, **kwargs),
		results,
		lambda box: box.name)
	return results


def sort_bounds(bounds, **kwargs):
	boxes = [Box(i, *args) for i, args in enumerate(bounds)]
	return _reading_order(boxes, **kwargs)


def sort_blocks(blocks, **kwargs):
	return _reading_order([
		Box(block, *block.polygon.bounds) for block in blocks],
		**kwargs)


def bounds_order(bounds, **kwargs):
	boxes = []

	for name, (minx, miny, maxx, maxy) in bounds:
		boxes.append(Box(name, minx, miny, maxx, maxy))

	return _reading_order(boxes, **kwargs)


def polygon_order(polygons, fringe, **kwargs):
	boxes = []

	for name, polygon in polygons:
		minx, miny, maxx, maxy = inset_bounds(
			polygon.bounds, fringe)
		boxes.append(Box(name, minx, miny, maxx, maxy))

	return _reading_order(boxes, **kwargs)
