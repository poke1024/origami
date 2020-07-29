#!/usr/bin/env python3

import imghdr
import click
import PIL.Image
import json
import math
import cv2
import collections
import shapely.ops
import shapely.geometry
import numpy as np

from pathlib import Path
from PySide2 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output, Annotation
from origami.core.page import Page
from origami.batch.annotate.utils import render_contours, render_lines, render_paths
from origami.batch.core.utils import TableRegionCombinator


def linestrings(geom):
	if geom is None:
		return
	if geom.geom_type == 'LineString':
		yield geom
	elif geom.geom_type in ('MultiLineString', 'GeometryCollection'):
		for g in geom.geoms:
			for ls in linestrings(g):
				yield ls
	else:
		raise RuntimeError("unexpected geom_type %s" % geom.geom_type)


def cond_table_data(data):
	data2 = dict()
	for k, v in data.items():
		k = tuple(k.split("/"))
		k2 = k[:-1] + (k[-1].split(".")[0],)
		data2["/".join(k2)] = v
	return data2


def column_paths(shape, x):
	if shape.is_empty:
		return
	ext_shape = shape.buffer(10)
	_, miny, _, maxy = shape.bounds
	line = shapely.geometry.LineString([[x, miny - 1], [x, maxy + 1]])
	for geom in linestrings(line.intersection(ext_shape)):
		yield list(geom.coords)


def divider_paths(shape, y):
	if shape.is_empty:
		return
	ext_shape = shape.buffer(10)
	minx, _, maxx, _ = shape.bounds
	line = shapely.geometry.LineString([[minx - 1, y], [maxx + 1, y]])
	for geom in linestrings(line.intersection(ext_shape)):
		yield list(geom.coords)


class DebugLayoutProcessor(Processor):
	def __init__(self, options):
		super().__init__(options, needs_qt=True)
		self._options = options
		self._overwrite = self._options["overwrite"]
		self._scale = max(0, min(2, self._options["scale"]))
		self._font_scale = max(0, min(100, self._options["font_scale"]))

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("warped", Input(Artifact.SEGMENTATION)),
			("reliable", Input(
				Artifact.CONTOURS, Artifact.LINES,
				Artifact.TABLES, Artifact.ORDER,
				stage=Stage.RELIABLE)),
			("output", Output(Annotation("layout"))),
		]

	def process(self, page_path: Path, warped, reliable, output):
		combinator = TableRegionCombinator(reliable.regions.by_path.keys())
		contours = combinator.contours_from_blocks(reliable.regions.by_path)

		rendered_contours = contours
		rendered_lines = []

		table_data = reliable.tables

		if self._options["label"] == "order":
			xycut_data = reliable.order
			order = [tuple(p.split("/")) for p in xycut_data["orders"]["*"]]

			unsplit_contour_paths = set([x for x in order if len(x) == 3])
			rendered_contours = dict(
				(k, v) for k, v in contours.items()
				if k in unsplit_contour_paths)

			rendered_line_paths = set([x for x in order if len(x) == 4])
			rendered_lines = dict(
				(k, v) for k, v in reliable.lines.by_path.items()
				if k in rendered_line_paths)

			order = dict(
				(path, i)
				for i, path in enumerate(order))

			def get_label(path):
				i = order.get(path)
				if i is not None:
					return path[:2], i + 1
				else:
					return path[:2], None

			def get_contour_label(block_path):
				return get_label(block_path)

			def get_line_label(line_path):
				return get_label(line_path)

		elif self._options["label"] == "id":
			labels = dict((x, int(x[2])) for x in contours.keys())

			def get_contour_label(block_path):
				return block_path[:2], labels.get(block_path)

			def get_line_label(x):
				return None
		else:
			raise ValueError(self._options["label"])

		page = reliable.page
		predictors = warped.predictors
		scale = self._scale
		font_scale = self._font_scale

		width, height = page.dewarped.size
		im = page.dewarped
		if scale != 1:
			pixels = cv2.resize(
				np.array(im),
				(int(width * scale), int(height * scale)),
				interpolation=cv2.INTER_AREA)
			im = PIL.Image.fromarray(pixels)

		qt_im = ImageQt(im)
		pixmap = QtGui.QPixmap.fromImage(qt_im)

		pixmap = render_contours(
			pixmap, rendered_contours, predictors,
			get_label=get_contour_label, alternate=False,
			scale=scale, font_scale=font_scale)

		pixmap = render_lines(
			pixmap, rendered_lines, predictors,
			get_label=get_line_label, show_vectors=True,
			scale=scale, font_scale=font_scale)

		columns = []
		for path, xs in cond_table_data(table_data["columns"]).items():
			path = tuple(path.split("/"))
			for x in xs:
				for coords in column_paths(contours[path], x):
					columns.append(coords)
		pixmap = render_paths(pixmap, columns, "blue", scale=scale)

		dividers = []
		for path, ys in cond_table_data(table_data["dividers"]).items():
			path = tuple(path.split("/"))
			for y in ys:
				for coords in divider_paths(contours[path], y):
					dividers.append(coords)
		pixmap = render_paths(pixmap, dividers, "magenta")

		output.annotation(pixmap.toImage())


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--label',
	type=str,
	default="order",
	help="How to label the block nodes.")
@click.option(
	'--scale',
	type=float,
	default=1,
	help="Scale of annotated image.")
@click.option(
	'--font-scale',
	type=float,
	default=1,
	help="Scale of label text of annotated image.")
@Processor.options
def debug_layout(data_path, **kwargs):
	""" Export annotate information on xycuts for all document images in DATA_PATH. """
	processor = DebugLayoutProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	debug_layout()
