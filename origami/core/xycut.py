# quite similar to the ideas described in:

# Ha, Jaekyu & Haralick, Robert & Phillips, Ihsin. (1995).
# Recursive X-Y cut using bounding boxes of connected components.
# Proceedings of the Third International Conference on Document Analysis
# and Recognition. 2. 952-955. 10.1109/ICDAR.1995.602059


import logging
import collections
import numpy as np

from itertools import chain


Candidate = collections.namedtuple(
	'Candidate', ['axis', 'x', 'score'])


Gap = collections.namedtuple(
	'Gap', ['minu', 'minv', 'maxu', 'maxv'])


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
	def __init__(self, objs, axis):
		xs = np.array([coords[:, axis] for coords in objs])

		c = np.hstack(xs)
		i = np.repeat(range(len(objs)), 2)

		s = np.argsort(c)

		self._objs = objs
		self._axis = axis

		self._x = c[s]
		self._label = i[s]

		self._min = np.min(xs, axis=-1)
		self._max = np.max(xs, axis=-1)

		ys = np.array([coords[:, 1 - axis] for coords in objs])
		self._ext = np.max(ys, axis=-1) - np.min(ys, axis=-1)

	def split_at(self, c):
		mask = self._x <= c

		# fix overlaps.
		a = set(self._label[mask])
		b = set(self._label[np.logical_not(mask)])

		for i in a & b:
			if abs(c - self._min[i]) < abs(c - self._max[i]):
				a.remove(i)
			else:
				b.remove(i)

		a = list(a)
		b = list(b)

		if not a:
			k = np.argmin([self._min[i] for i in b])
			a.append(b[k])
			del b[k]
		elif not b:
			k = np.argmax([self._max[i] for i in a])
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
				if n == 0:  # classic xy cut on whitespace.
					gap = Gap(
						minu=x0,
						minv=min(self._min[i0], self._min[i1]),
						maxu=x1,
						maxv=max(self._max[i0], self._max[i1]))
					yield Candidate(
						self, x0, score(self._axis, gap))
				elif n >= 1:
					err = 0
					for j in active_set.keys():
						err += self._ext[j] * min(
							abs(x0 - self._min[j]),
							abs(x0 - self._max[j]))
					yield Candidate(
						self, x0, -err)


def score_area(axis, gap):
	return (gap.maxu - gap.minu) * (gap.maxv - gap.minv)


def score_u(axis, gap):
	return gap.maxu - gap.minu


def score_v(axis, gap):
	return gap.maxv - gap.minv


default_scores = dict(
	area=score_area,
	u=score_u,
	v=score_v)


class XYCut:
	def __init__(self, objs, score="v", eps=5):
		if isinstance(score, str):
			score = default_scores[score]
		if len(objs) >= 2:
			coords = [np.array(o.coords, dtype=np.float64) for o in objs]
			lcs = [Coordinates(coords, axis) for axis in (0, 1)]
			splits = list(chain(*[lc.candidate_splits(score=score, eps=eps) for lc in lcs]))
		else:
			splits = None

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
