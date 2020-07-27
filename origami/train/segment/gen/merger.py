import numpy as np
import collections
import shapely.strtree


def _gen_pts(segments):
	for i, s in enumerate(segments):
		a, b = s.endpoints
		if any(a != b):
			yield i, a
			yield i, b


class Merger:
	strategies = dict(
		by_distance=lambda m: m.merge_by_endpoints,
		parallel=lambda m: m.merge_parallel,
		by_length=lambda m: m.filter_by_length,
		by_quality=lambda m: m.filter_by_quality
	)

	def __init__(self, merge_spec, label_set, labels, segments):
		from .segments import SegmentJoiner

		self._label_set = label_set
		self._labels = labels.copy()  # will get modified
		self._segments = list(segments)
		self._join_regions = np.zeros(labels.shape, dtype=np.uint8)
		self._segment_joiner = SegmentJoiner(merge_spec["joiner"], label_set)

		for stage in merge_spec["merge"]:
			f = Merger.strategies[stage["strategy"]](self)
			f(**stage["args"])

	@property
	def segments(self):
		return self._segments

	def filter_by_quality(self, max_error, min_length):
		scale = min(*self._labels.shape)
		self._segments = [
			s for s in self._segments
			if s.error < max_error and s.length >= min_length * scale]

	def _debug_join_region(self, dominant_label, mask):
		# for debugging
		if self._join_regions is not None:
			self._join_regions[mask] = dominant_label.index

	@property
	def join_regions(self):
		return Annotations(self._join_regions)

	def merge_by_endpoints(self, distances):
		# try to merge by nearest endpoints.

		from .segments import JoinResult

		# 10 max distance for TABCOL fixes table spilling over
		# "Erze." text in SNP2436020X-19100601-1-0-0-0.03

		pts = list(_gen_pts(self._segments))

		candidates = []
		for i, (ia, a) in enumerate(pts):
			for j, (ib, b) in enumerate(pts[:i]):
				if ia == ib:
					continue
				seg_a = self._segments[ia]
				seg_b = self._segments[ib]
				if seg_a.dominant_label != seg_b.dominant_label:
					continue
				d = np.linalg.norm(a - b)
				# print("dist", ia, ib, d)
				if d < distances.get(seg_a.dominant_label.name, 0):
					candidates.append((d, ia, a, ib, b))
		candidates = sorted(candidates, key=lambda c: c[0])

		for _, ia, a, ib, b in candidates:
			seg_a = self._segments[ia]
			seg_b = self._segments[ib]

			if seg_a is seg_b:
				continue

			# print("checking", a, b)

			ea = seg_a.endpoint_i(a)
			if ea is None:  # already merged?
				continue

			eb = seg_b.endpoint_i(b)
			if eb is None:  # already merged?
				continue

			if ea == eb:
				# never merge two start points or two end points. we assume
				# our segment endpoint indices are always ordered from left
				# to right (for H) or top to bottom (for V, TABCOL).
				continue

			# if ea == eb:
			#    print("NOPE", a, b)
			# assert ea != eb

			# print("!", a.name, b.name)
			result, s = self._segment_joiner.join(
				self._labels, seg_a, seg_b, (ea, eb), self._debug_join_region)

			if result == JoinResult.OK and s is not None:
				# patch labels.
				self._labels[s.mask] = s.dominant_label.index

				self._segments[ia] = s
				self._segments[ib] = s

				assert all(seg_a.endpoints[1 - ea] == s.endpoints[1 - ea])
				assert all(seg_b.endpoints[1 - eb] == s.endpoints[1 - eb])

				assert s.endpoint_i(a) is None
				assert s.endpoint_i(b) is None

				# print("merged", ia, ib, s.endpoints)

		segs = dict()
		for s in self._segments:
			segs[id(s)] = s
		self._segments = list(segs.values())

	def merge_parallel(self, overlap_buffer, close_distance):
		# now merge parallel segments that overlap or are very near
		# (see overlap_buffer).

		from .segments import Segment

		while True:
			segs = collections.defaultdict(list)
			for s in self._segments:
				segs[s.path.bounds].append(s)

			tree = shapely.strtree.STRtree([s.path for s in self._segments])
			merged = set()
			added = []

			for s in self._segments:
				if s.name in merged:
					continue
				for other_path in tree.query(s.path.buffer(overlap_buffer)):
					if not other_path.equals(s.path):
						for t in segs[other_path.bounds]:
							if t.name in merged:
								continue
							if s.dominant_label != t.dominant_label:
								continue
							if Segment.parallel_and_close(s, t, close_distance):
								added.append(Segment.from_mask(
									self._label_set,
									self._labels,
									np.logical_or(s.mask, t.mask),
									"%s+%s" % (s.name, t.name)))
								merged.add(s.name)
								merged.add(t.name)

			old_n = len(self._segments)
			self._segments = [s for s in self._segments if s.name not in merged] + added
			if old_n == len(self._segments):
				break

	def filter_by_length(self, lengths):
		# get rid of remaining noise.
		self._segments = [
			s for s in self._segments
			if s.length >= lengths.get(s.dominant_label.name, 0)]

	def filter_by_region(self, region):
		self._segments = list(filter(lambda s: not s.path.disjoint(region), self._segments))
