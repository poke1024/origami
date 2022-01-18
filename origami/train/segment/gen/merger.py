import numpy as np
import collections
import shapely.strtree
import intervaltree
import tesserocr
import PIL.Image

from origami.train.segment.gen.masks import polygons_to_mask


def _gen_pts(segments):
	for i, s in enumerate(segments):
		a, b = s.endpoints
		if any(a != b):
			yield i, a
			yield i, b


class SegmentMerger:
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
		self._segment_joiner = SegmentJoiner(
			merge_spec["segments"]["obstacles"], label_set)

		for stage in merge_spec["segments"]["pipeline"]:
			f = SegmentMerger.strategies[stage["strategy"]](self)
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


class PolygonV:
	def __init__(self, polygons):
		self._tree = intervaltree.IntervalTree()

		self._polygons = polygons

		for i, polygon in enumerate(polygons):
			_, miny, _, maxy = polygon.bounds
			self._tree[miny:maxy + 1] = i

	def query(self, polygon):
		_, miny, _, maxy = polygon.bounds
		min_y_overlap = 3
		# set min_y_overlap to not merge "tüchtiger Beamter" with top text on SNP2436020X-19100601-0-0-0-0.10

		ivs = sorted(
			self._tree.overlap(miny + min_y_overlap, maxy + 1 - min_y_overlap),
			key=lambda iv: self._polygons[iv.data].bounds[0])

		return list(map(lambda iv: iv.data, ivs))

	def remove(self, index):
		_, miny, _, maxy = self._polygons[index].bounds
		self._tree.removei(miny, maxy + 1, index)


class HMerger:
	def __init__(self, polygons, stoppers):
		self._polygons = polygons
		self._stop = shapely.strtree.STRtree(stoppers)

		# we sometimes touch text regions which prevent merging, that's
		# why we erode a bit here to allow merging to allow touching
		# text regions.
		# -20 derived from tests on SNP2436020X-19100601-0-0-0-0.09.
		self._erosion = 20

	@property
	def polygons(self):
		return self._polygons

	def _removed_polygon(self, polygon):
		pass

	def _added_polygon(self, polygon):
		pass

	def _can_merge(self, polygon1, polygon2):
		centroid1 = polygon1.centroid.coords[0]
		centroid2 = polygon2.centroid.coords[0]
		connection = shapely.geometry.LineString([centroid1, centroid2])

		if self._stop.query(connection):
			return False
		else:
			return True

	def _can_merge_into(self, polygon):
		if self._stop.query(polygon):
			return False
		else:
			return True

	def _sort_by_x(self):
		self._polygons = sorted(self._polygons, key=lambda p: p.bounds[0])

	def _begin_merge_step(self):
		self._sort_by_x()

	def _merge_step(self):
		self._begin_merge_step()

		tree = PolygonV(self._polygons)

		was_merged = np.zeros((len(self._polygons),), dtype=np.bool)
		merged_polygons = []

		for i, polygon1 in enumerate(self._polygons):
			if was_merged[i]:
				continue

			tree.remove(i)

			for j in tree.query(polygon1):
				assert j != i

				polygon2 = self._polygons[j]

				if not self._can_merge(polygon1, polygon2):
					continue

				union = shapely.ops.unary_union([polygon1, polygon2]).convex_hull

				if not self._can_merge_into(union.buffer(-self._erosion)):
					continue

				# good to go.
				was_merged[j] = True
				tree.remove(j)

				self._removed_polygon(polygon1)
				self._removed_polygon(polygon2)
				self._added_polygon(union)

				# update polygon1.
				polygon1 = union

			merged_polygons.append(polygon1)

		success = len(merged_polygons) < len(self._polygons)
		self._polygons = merged_polygons
		return success

	def merge(self):
		while self._merge_step():
			pass

		return self._polygons


class HTextMerger(HMerger):
	def __init__(self, unbinarized, polygons, stoppers):
		super().__init__(polygons, stoppers)
		self._unbinarized = unbinarized
		self._ocr = dict()
		self._unmerged = None
		self._debug = False

	def _line_height(self, polygon):
		key = tuple(polygon.centroid.coords[0])
		if key not in self._ocr:
			mask = polygons_to_mask(self._unbinarized.shape, [polygon])

			minx, miny, maxx, maxy = polygon.bounds
			minx, miny = np.floor(np.array([minx, miny])).astype(np.int32)
			maxx, maxy = np.ceil(np.array([maxx, maxy])).astype(np.int32)

			pixels = self._unbinarized[miny:maxy, minx:maxx]
			mask = mask[miny:maxy, minx:maxx]
			pixels[np.logical_not(mask)] = 255

			with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_BLOCK) as api:
				api.SetImage(PIL.Image.fromarray(pixels, "L"))

				heights = []
				for i, data in enumerate(api.GetTextlines()):
					bbox = data[1]
					heights.append(bbox["h"])

				if heights:
					n_lines = len(heights)
					lh = np.min(heights)
				else:
					lh = maxy - miny
					n_lines = 1

				if self._debug:
					api.Recognize()

					ri = api.GetIterator()
					level = tesserocr.RIL.TEXTLINE

					text = ""
					# lines = []
					for r in tesserocr.iterate_level(ri, level):
						# baseline = r.Baseline(level)
						# if baseline:
						#	p1, p2 = baseline
						#	lines.append(shapely.geometry.LineString([p1, p2]))

						try:
							text += r.GetUTF8Text(level) + " "
						except RuntimeError:
							pass

				# print("txt", text.strip(), "lh", lh, "#", n_lines)
				else:
					text = ""

			self._ocr[key] = (n_lines, lh, text)

		return self._ocr[key]

	def _begin_merge_step(self):
		super()._begin_merge_step()
		self._unmerged = shapely.strtree.STRtree(self._polygons)

	def _removed_polygon(self, polygon):
		super()._removed_polygon(polygon)
		key = tuple(polygon.centroid.coords[0])
		if key in self._ocr:
			del self._ocr[key]

	def _can_merge(self, polygon1, polygon2):
		if not super()._can_merge(polygon1, polygon2):
			return False

		n_lines1, lh1, text1 = self._line_height(polygon1)
		n_lines2, lh2, text2 = self._line_height(polygon2)

		if lh1 is None or lh2 is None:
			# never merge if no text is detected.
			return False

		if n_lines1 == 1 and n_lines2 == 1:
			y1 = polygon1.centroid.coords[0][1]
			y2 = polygon2.centroid.coords[0][1]
			if abs(y1 - y2) > min(lh1, lh2) * 0.5:
				return False

		minx, miny, maxx, maxy = polygon1.bounds
		dx1 = maxx - minx

		minx, miny, maxx, maxy = polygon2.bounds
		dx2 = maxx - minx

		# 1.5 based on SNP2436020X-19100601-0-0-0-0.10 and SNP2436020X-19100601-0-0-0-0.11.
		# and SNP2436020X-19100601-1-0-0-0.03.
		max_dy_ratio = 1.5
		dy_ratio = max(lh1 / lh2, lh2 / lh1)

		if dy_ratio > max_dy_ratio:
			if self._debug:
				print("reject by max_dy_ratio", dy_ratio, lh1, lh2, text1, text2)
			return False

		max_dx = 0.25

		if polygon1.distance(polygon2) > max_dx * max(dx1, dx2):
			if self._debug:
				print("reject by distance", text1, text2)
			return False

		return True

	def _can_merge_into(self, polygon):
		if not super()._can_merge_into(polygon):
			return False

		# if merging merges more than two unmerged segments,
		# something went wrong. we might have skipped an
		# intermediate polygon that now gets merged.
		if len(self._unmerged.query(polygon)) > 2:
			return False

		return True
