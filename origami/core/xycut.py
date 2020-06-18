# quite similar to the ideas described in:

# Ha, Jaekyu & Haralick, Robert & Phillips, Ihsin. (1995).
# Recursive X-Y cut using bounding boxes of connected components.
# Proceedings of the Third International Conference on Document Analysis
# and Recognition. 2. 952-955. 10.1109/ICDAR.1995.602059


import logging
import collections
import numpy as np

from itertools import chain


SplitCandidate = collections.namedtuple(
	'SplitCandidate', ['axis', 'overlap', 'x', 'gap'])


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


class LabeledCoordinates:
	def __init__(self, objs, axis):
		self._sizes = np.array([len(coords) for coords in objs])

		c = np.hstack([coords[:, axis] for coords in objs])
		i = np.repeat(range(len(objs)), self._sizes)

		s = np.argsort(c)

		self._objs = objs
		self._axis = axis

		self._x = c[s]
		self._label = i[s]

	def split_at(self, c, has_overlaps=False):
		mask = self._x <= c

		a = np.unique(self._label[mask])
		b = np.unique(self._label[np.logical_not(mask)])

		if has_overlaps:
			a = set(a)
			b = set(b)

			for i in a & b:
				x0, x1 = self._objs[i][:, self._axis]
				if abs(x0 - c) < abs(x1 - c):
					a.remove(i)
				else:
					b.remove(i)

			a = list(a)
			b = list(b)

		return a, b

	def items(self):
		return zip(self._x, self._label)

	def candidate_splits(self):
		counts = collections.defaultdict(int)
		sizes = self._sizes

		split_after = None

		for c, i in self.items():

			if split_after is not None and len(counts) != 1:
				yield SplitCandidate(
					self, len(counts), split_after, c - split_after)

			counts[i] += 1
			if counts[i] == sizes[i]:
				del counts[i]

			split_after = c


class XYCut:
	def __init__(self, objs):
		coords = [np.array(o.coords, dtype=np.float64) for o in objs]
		lcs = [LabeledCoordinates(coords, axis) for axis in (0, 1)]
		splits = list(chain(*[lc.candidate_splits() for lc in lcs]))

		if not splits:
			self._split = None
			self._axis = None
			self._x = None
		else:
			best = sorted(splits, key=lambda x: (-x.overlap, x.gap))[-1]

			if best.overlap > 0:  # go for smallest gap step instead.
				best = sorted(splits, key=lambda x: (-x.overlap, -x.gap))[-1]

			ia, ib = best.axis.split_at(best.x, best.overlap > 0)
			self._split = [objs[i] for i in ia], [objs[i] for i in ib]

			self._axis = lcs.index(best.axis)
			self._x = best.x

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


def _rxy_cut(boxes):
	if len(boxes) <= 1:
		return [*boxes], []

	cut = XYCut(boxes)
	if not cut.valid:
		return [*boxes], []

	if max(len(cut[0]), len(cut[1])) < len(boxes):
		return list(map(_rxy_cut, cut))
	else:
		logging.info("aborting _rxy_cut (%d elements)." % len(boxes))
		return [*boxes], []


def _flatten(boxes, leafs):
	if isinstance(boxes, Box):
		leafs.append(boxes)
	else:
		for x in boxes:
			_flatten(x, leafs)


def reading_order(bounds):
	boxes = [Box(i, *args) for i, args in enumerate(bounds)]
	leafs = []
	_flatten(_rxy_cut(boxes), leafs)
	return [box.name for box in leafs]


def sort_blocks(blocks):
	order = reading_order([
		block.polygon.bounds for block in blocks])
	return [blocks[i] for i in order]
