"""
BBZ-specific annotation postprocessing. We are operating on
a deskewed image.
"""

import numpy as np
import skimage
import cv2
import shapely.geometry
import skimage.morphology
import collections

from origami.train.segment.gen.geometry import (
	regions_to_convex_hull, merge_convex_all, Simplifier,
	contours, convex_contours, convex_hull)
from origami.train.segment.gen.masks import mask_to_polygons, polygons_to_mask
from origami.train.segment.gen.merger import HMerger, HTextMerger
from origami.train.segment.gen.annotations import Annotations


# --------------------------------------------------------------------------------
# utility functions.
# --------------------------------------------------------------------------------


def v_separator_stopper(gen, kernel=(10, 30), tabcol=True):
	return gen.stops.mask_by_name(
		*(["V"] + (["TABCOL"] if tabcol else []))).astype(np.uint8)


def separator_stopper(gen, kernel=(10, 10), tabcol=True):
	return np.logical_or(
		gen.stops.mask_by_name("H", "BORDER", "H_SM"),
		v_separator_stopper(gen, tabcol=tabcol)).astype(np.uint8)


def text_stopper(gen, kernel=(7, 5), iterations=3):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)

	stopper = gen.master.mask_by_name(
		"FRAKTUR_BG", "FRAKTUR_SM",
		"ANTIQUA_BG", "ANTIQUA_SM").astype(np.uint8)

	for _ in range(iterations):
		stopper = cv2.morphologyEx(
			stopper, cv2.MORPH_CLOSE, kernel, iterations=1)

		stopper = cv2.dilate(stopper, kernel, iterations=1)

	return stopper


def table_stopper(gen, kernel=(5, 3), iterations=3):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)

	stopper = gen.master.mask_by_name("TABTXT").astype(np.uint8)

	for _ in range(iterations):
		stopper = cv2.morphologyEx(
			stopper, cv2.MORPH_CLOSE, kernel, iterations=1)

		stopper = cv2.dilate(stopper, kernel, iterations=1)

	return stopper


def erode(h):
	return h.buffer(-2)


def illustrations_stopper(gen, macro_regions):
	background = gen.master.mask_by_name("ILLUSTRATION")
	background = cv2.dilate(
		background.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)), iterations=1)
	# kernel, see SNP2436020X-19200601-1-0-0-0.03.
	background = np.logical_and(background, np.logical_not(macro_regions))
	return shapely.ops.cascaded_union(
		mask_to_polygons(background, convex_hulls=True))


# --------------------------------------------------------------------------------
# table region extraction.
# --------------------------------------------------------------------------------

def table_regions_at_iterations(gen, kernel=(10, 3), iterations=(3, 5)):
	despeckle_tol = 15

	table_mask = gen.master.mask_by_name("TABTXT", "TABCOL")
	table_mask = skimage.morphology.remove_small_objects(
		table_mask.astype(np.bool), despeckle_tol)
	table_mask = table_mask.astype(np.uint8)

	stopper = np.logical_not(np.logical_or(
		separator_stopper(gen, tabcol=False), text_stopper(gen)))
	results = []

	extend_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
	for i in range(max(iterations)):
		table_mask = cv2.morphologyEx(
			table_mask, cv2.MORPH_CLOSE, extend_kernel, iterations=1)
		table_mask = np.logical_and(table_mask, stopper).astype(np.uint8)

		table_mask = cv2.dilate(table_mask, extend_kernel, iterations=1)
		table_mask = np.logical_and(table_mask, stopper).astype(np.uint8)

		if (i + 1) in iterations:
			results.append(regions_to_convex_hull(table_mask).astype(np.uint8))

	return tuple(results)


def table_polygons(gen, kernel=(6, 5)):
	# kernel = (6, 5) fixes missed left bottom " in SNP2436020X-19100601-0-0-0-0.13
	# and fixes non-contiguous table regions in SNP2436020X-19100601-1-0-0-0.03
	# and avoids spilling over v sep in SNP2436020X-19000601-0-0-0-0.09

	# mini_regions = self._table_regions_at_iterations((3, 3), (1,))[0]

	micro_regions, macro_regions = table_regions_at_iterations(gen, kernel, (2, 5))
	# don't do more than 5 iterations here, otherwise tables will merge over text in
	# SNP2436020X-19100601-0-0-0-0.14

	background0 = text_stopper(gen, kernel=(7, 3), iterations=1)
	# iterations=1 tuned via SNP2436020X-19100601-0-0-0-0.13.
	# kernel, see SNP2436020X-19200601-1-0-0-0.03.

	labels, num = skimage.morphology.label(
		macro_regions.astype(np.bool), return_num=True)

	simplify = Simplifier(simplify=0)

	# detect border cases like bottom table in 2436020X_1925-02-27_70_98_006.
	tree = shapely.strtree.STRtree([
		s.path for s in gen.segments if s.dominant_label.name == "V"])

	hulls = []

	# tables = np.zeros(self._annotations.shape, dtype=np.bool)
	for i in range(1, num + 1):
		added = np.logical_and(labels == i, micro_regions)

		if not added.any():
			continue

		added = skimage.morphology.convex_hull_image(added)
		added = np.logical_and(added, np.logical_not(background0))
		hull = convex_hull(simplify(contours(added)))

		segment_by = None
		for s in tree.query(hull):
			if hull.buffer(-5).contains(shapely.geometry.Point(*s.centroid.coords)):
				segment_by = s
				break

		if segment_by:
			added = np.logical_and(labels == i, micro_regions)
			added = np.logical_and(added, np.logical_not(background0))
			for c in convex_contours(added):
				hulls.append(c)
		else:
			hulls.append(hull)

	return merge_convex_all(hulls)


# --------------------------------------------------------------------------------
# text region extraction.
# --------------------------------------------------------------------------------


def text_region(gen, text_label, kernel=(10, 3), iterations=2, close_iterations=1):
	despeckle_tol = 15

	no_text_mask = np.logical_or(
		separator_stopper(gen), table_stopper(gen))

	# text region.
	text_mask = gen.master.mask_by_name(text_label)
	text_mask = skimage.morphology.remove_small_objects(text_mask.astype(np.bool), despeckle_tol)
	text_mask = text_mask.astype(np.uint8)

	extend_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
	stopper = np.logical_not(no_text_mask)

	for _ in range(iterations):
		text_mask = cv2.dilate(
			text_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel), iterations=1)

		text_mask = cv2.morphologyEx(
			text_mask, cv2.MORPH_CLOSE, extend_kernel, iterations=close_iterations)
		text_mask = np.logical_and(text_mask, stopper).astype(np.uint8)

		text_mask = np.logical_and(text_mask, stopper).astype(np.uint8)

	text_mask = np.logical_and(text_mask, stopper)

	return text_mask.astype(np.uint8)


def text_polygons(gen):
	text_labels = ("FRAKTUR_BG", "FRAKTUR_SM", "ANTIQUA_BG", "ANTIQUA_SM")

	results = []
	for text_label in text_labels:
		other_text_labels = [l for l in text_labels if l != text_label]

		background0 = gen.master.mask_by_name("ILLUSTRATION", "TABTXT", *other_text_labels)
		background0 = cv2.dilate(
			background0.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)), iterations=1)

		mini_regions = text_region(gen, text_label, (8, 2), 1)
		micro_regions = text_region(gen, text_label, (8, 3), 1)
		macro_regions = text_region(gen, text_label, (8, 3), 3, close_iterations=3)
		# (8, 3), 3 ensures that "Handelsnachrichten." head stays separated on SNP2436020X-19100601-0-0-0-0.12
		# 8 (< 9!) ensures that "Berliner Börsenzeitung" stays horizontally separated, SNP2436020X-19100601-1-0-0-0.00
		# FIXME: (9, 3) would connect "Ein Sommernachtstraum" in SNP2436020X-19000601-0-0-0-0.13.

		# kernel, see SNP2436020X-19200601-1-0-0-0.03.
		macro_regions = np.logical_and(macro_regions, np.logical_not(background0))

		labels, num = skimage.morphology.label(macro_regions.astype(np.bool), return_num=True)
		#region = np.zeros(self._annotations.shape, dtype=np.bool)
		hulls = []

		simplify = Simplifier(simplify=0)

		# detect border cases like "Schiller-Theater" in SNP2436020X-19100601-0-0-0-0.11.
		tree = shapely.strtree.STRtree([s.path for s in gen.segments if s.dominant_label.name == "V"])

		for i in range(1, num + 1):
			mask = np.logical_and(labels == i, micro_regions)
			if not mask.any():
				continue

			hull = convex_hull(simplify(contours(mask)))

			segment_by = None
			for s in tree.query(hull):
				if hull.buffer(-5).contains(shapely.geometry.Point(*s.centroid.coords)):
					segment_by = s
					break

			if segment_by:
				micros = list(simplify(convex_contours(
					np.logical_and(labels == i, mini_regions))))

				region = collections.defaultdict(list)
				coords = np.array(segment_by.coords)
				x0 = np.median(coords[:, 0])
				y0 = np.min(coords[:, 1])
				for m in micros:
					x, y = m.centroid.coords[0]
					if y < y0:
						region["top"].append(m)
					elif x < x0:
						region["left"].append(m)
					else:
						region["right"].append(m)
				for k, v in region.items():
					hulls.append(shapely.ops.cascaded_union(v).convex_hull)
			else:
				hulls.append(hull)

		hulls = merge_convex_all(hulls)
		hulls = [erode(h) for h in hulls]
		results.extend(hulls)

	#region = polygons_to_mask(mask.shape, hulls)

	illustrations = illustrations_stopper(gen, macro_regions)
	#region = np.logical_and(region, np.logical_not(illustrations))

	# remove_small_objects fixes noise in SNP2436020X-19200601-1-0-0-0.03.
	#region = skimage.morphology.remove_small_objects(region, 100)

	#return region.astype(np.uint8)

	result = shapely.ops.cascaded_union(results).difference(illustrations)
	if result.geom_type == "Polygon":
		return [result]
	else:
		polygons = list(result.geoms)
		return [p for p in polygons if p.geom_type == "Polygon" and p.area > 100]


# --------------------------------------------------------------------------------

def generate(gen, pixels):
	""""
	takes a deskewed image (pixels) and an AnnotationGenerator (gen) and
	produces a cleaned up annotation with dilated regions.
	"""

	tables = table_polygons(gen)
	text = text_polygons(gen)
	seps = [s.path for s in gen.segments if s.dominant_label.name != "TABCOL"]

	merger = HMerger(
		tables,
		text + seps)
	tables = merger.merge()

	merger = HTextMerger(
		gen.deskewed(pixels),
		text,
		tables + [s.path for s in gen.segments])
	text = merger.merge()

	lset = gen.label_set
	shape = gen.master.shape
	labels = lset.labels_by_name

	optimized_mask = np.zeros(shape, dtype=np.uint8)
	optimized_mask.fill(labels["BACKGROUND"].index)

	text_mask = polygons_to_mask(shape, text)
	tables_mask = polygons_to_mask(shape, tables)

	optimized_mask[tables_mask] = labels["TABTXT"].index
	optimized_mask[text_mask] = labels["ANTIQUA_SM"].index

	slabels = gen.master.separator_labels
	slabels_mask = slabels != labels["BACKGROUND"].index
	optimized_mask[slabels_mask] = slabels[slabels_mask]

	clabels = gen.skewed_annotation.labels
	ann = Annotations(lset, optimized_mask).transform(gen.deskewing.inverse)
	ann.mutable_labels[clabels == labels["BORDER"].index] = labels["BORDER"].index
	ann.mutable_labels[clabels == labels["ILLUSTRATION"].index] = labels["ILLUSTRATION"].index

	return ann
