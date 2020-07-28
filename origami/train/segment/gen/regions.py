import numpy as np
import cv2
import shapely
import shapely.strtree
import shapely.ops
import math

from .skew import estimate_skew


class AnnotationsGenerator:
	def __init__(self, label_set, merge_spec, ann, segments):
		from . import transform

		settings = merge_spec["segments"]["painter"]

		self._label_set = label_set
		self._segments = segments
		self._skewed_ann = ann
		self._master = self._generate(
			segment_thickness_scale=settings["segment_thickness"])

		self._skew = estimate_skew(segments)

		self._deskewing = transform.Rotate(
			reversed(self._master.shape), -self._skew)

		extend = dict()
		for k, v in settings["segment_extend"].items():
			extend[label_set.label_from_name(k)] = v

		self._master = self._master.transform(self._deskewing)
		self._stops = self._generate(
			segment_thickness_scale=settings["segment_thickness"],
			segment_extend_amount=extend).transform(self._deskewing)

		self._deskewed_segments = [s.transform(self._deskewing) for s in self._segments]

	@property
	def deskewing(self):
		return self._deskewing

	@property
	def master(self):
		return self._master

	@property
	def stops(self):
		return self._stops

	@property
	def segments(self):
		return self._deskewed_segments

	@property
	def label_set(self):
		return self._label_set

	def deskewed(self, pixels):
		return cv2.warpAffine(
			pixels,
			self.deskewing.matrix,
			self.deskewing.target_size,
			flags=cv2.INTER_AREA)

	@property
	def skewed_annotation(self):
		return self._skewed_ann

	def _generate(self, segment_thickness_scale=1, segment_extend_amount=None):
		data = self.skewed_annotation.non_separator_labels

		line_shift = 0
		line_scale = 1 << line_shift

		def draw_polyline(pts, label, thickness):
			for a, b in zip(pts, pts[1:]):
				a = np.round(a * line_scale).astype(np.int32)
				b = np.round(b * line_scale).astype(np.int32)
				cv2.line(data, tuple(a), tuple(b), label.index, thickness=thickness, shift=line_shift)

		for i, s in enumerate(self._segments):

			if segment_extend_amount and s.dominant_label in segment_extend_amount:
				s = s.extend_by(segment_extend_amount[s.dominant_label])

				# clip this against all non-extended labels.
				for j, t in enumerate(self._segments):
					if i != j:
						try:
							shapes = shapely.ops.split(s.path, t.path)
							if len(shapes.geoms) == 2:
								k = np.argmax([geom.length for geom in shapes.geoms])
								s._path = shapes.geoms[k]  # HACK
						except ValueError:  # error in split on full overlap
							pass  # ignore

			thickness = max(2, int(math.floor(s.thickness * segment_thickness_scale)))
			draw_polyline(s.path.coords, s.dominant_label, thickness)

		from .annotations import Annotations
		return Annotations(self._label_set, data)


'''
class Regions:
	def __init__(self, label_set, clabels, segments):
		gen = AnnotationsGenerator(label_set, clabels, segments)

		self._segments = gen.segments

		self._shape = self._slabels.shape
		self._deskewing = gen.deskewing

		self._clabels = clabels
		self._label_set = label_set

	@property
	def deskewing(self):
		return self._deskewing

	@property
	def input(self):
		return self._morph.input

	@property
	def to_deskewed_segments(self):  # for debugging
		pixels = self._morph.input.labels.copy()

		def draw_polyline(pts, label, thickness):
			for a, b in zip(pts, pts[1:]):
				a = np.round(a ).astype(np.int32)
				b = np.round(b).astype(np.int32)
				cv2.line(pixels, tuple(a), tuple(b), label, thickness=thickness, shift=0)

		for s in self._segments:
			draw_polyline(list(s.path.coords), s.dominant_label, 16)

		return Annotations(self._label_set, pixels)

	def to_text_boxes(self):
		mask = np.zeros(self._shape, dtype=np.uint8)
		mask.fill(int(Label.BACKGROUND))

		for i, polygon in enumerate(self._text):
			minx, miny, maxx, maxy = polygon.bounds

			p1 = np.array([minx, miny]).astype(np.int32)
			p2 = np.array([maxx, maxy]).astype(np.int32)
			cv2.rectangle(mask, tuple(p1), tuple(p2), int(Label.ANTIQUA_SM), thickness=2)

			cv2.putText(
				mask, "(%d,%d)" % (minx, miny), (int(minx), int((miny + maxy) / 2)),
				cv2.FONT_HERSHEY_SIMPLEX, 1, int(Label.ANTIQUA_SM), 2)

		return Annotations(self._label_set, mask).transform(self.deskewing.inverse)
'''


'''
	@property
	def text_macro(self):
		return self._text_region((8, 3), 3, close_iterations=3)

	@property
	def text_background(self):
		background0 = self._annotations.mask(Label.ILLUSTRATION, Label.TABTXT)
		background0 = cv2.dilate(
			background0.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)), iterations=1)
		# kernel, see SNP2436020X-19200601-1-0-0-0.03.
		return np.logical_and(background0, np.logical_not(self.text_macro))
'''

