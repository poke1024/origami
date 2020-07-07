#!/usr/bin/env python3

import numpy as np
import multiprocessing.pool
import collections
import json
import click
import shapely.ops
import functools

from pathlib import Path
from functools import partial

import origami.core.binarize


def reliable_contours(all_contours, all_lines, min_confidence=0.5):
	block_lines = collections.defaultdict(list)
	for path, line in all_lines.items():
		if line.confidence > min_confidence:
			block_lines[path[:3]].append(line)

	reliable = dict()
	for path, lines in block_lines.items():
		hull = shapely.ops.cascaded_union([
			line.image_space_polygon for line in lines]).convex_hull
		geom = hull.intersection(all_contours[path])
		if geom.geom_type != "Polygon":
			geom = geom.convex_hull
		reliable[path] = geom

	# for contours, for which we have to lines at all, we keep
	# the contour as is.
	for k in set(all_contours.keys()) - set(reliable.keys()):
		reliable[k] = all_contours[k]

	return reliable


class LineExtractor:
	def __init__(self, tables, line_height, options, min_confidence=0.5):
		self._options = options
		self._line_height = line_height
		assert self._line_height is not None

		if self._options["binarize"].strip():
			self._binarizer = origami.core.binarize.from_string(
				self._options["binarize"])
		else:
			self._binarizer = None

		self._min_confidence = min_confidence

		self._columns = dict(
			(tuple(k.split("/")), xs)
			for k, xs in tables["columns"].items())

	@staticmethod
	def options(f):
		options = [
			click.option(
				'--binarize',
				type=str,
				default="",
				help="binarization algorithm to use (e.g. otsu), or empty if none"),
			click.option(
				'--do-not-dewarp',
				default=False,
				is_flag=True,
				help='do not dewarp line images'),
			click.option(
				'--do-not-deskew',
				default=False,
				is_flag=True,
				help='do not deskew line images')
		]
		return functools.reduce(lambda x, opt: opt(x), options, f)

	def _extract_line_image(self, item):
		line_path, line, column = item

		return line_path, line.image(
			target_height=self._line_height,
			column=column,
			dewarped=not self._options["do_not_dewarp"],
			deskewed=not self._options["do_not_deskew"],
			binarizer=self._binarizer)

	def _column_path(self, path, column):
		assert column >= 1
		predictor, label = path[:2]
		parts = path[2].split(".")
		if len(parts) != 4:
			raise RuntimeError("%s is not a valid table path" % str(path))
		block, division, _, _ = parts
		line = 1 + int(path[-1])  # index of line inside this block division
		grid = ".".join(map(str, (block, division, line, column)))
		return predictor, label, grid, str(0)

	def __call__(self, lines, ignored=None):
		if ignored is not None:
			lines = dict(
				(k, v) for k, v in lines.items()
				if not ignored(tuple(k[:2])))

		# this logic is intertwined with the logic in subdivide_table_blocks(). we
		# split table rows, if our table data says so.

		line_parts = []
		for path, line in lines.items():
			if line.confidence > self._min_confidence:
				line_columns = self._columns.get(path[:3])
				if line_columns is None:
					line_parts.append((path, line, None))
				else:
					line_columns = [None] + line_columns + [None]
					for i, (x0, x1) in enumerate(zip(line_columns, line_columns[1:])):
						line_parts.append((self._column_path(path, 1 + i), line, (x0, x1)))

		pool = multiprocessing.pool.ThreadPool(processes=8)
		return pool.map(self._extract_line_image, line_parts)
