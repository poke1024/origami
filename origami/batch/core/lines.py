import numpy as np
import multiprocessing.pool
import collections
import json
import shapely.ops

from pathlib import Path
from functools import partial

from origami.core.block import Binarizer


def reliable_contours(all_blocks, all_lines, min_confidence=0.5):
	block_lines = collections.defaultdict(list)
	for path, line in all_lines.items():
		if line.confidence > min_confidence:
			block_lines[path[:3]].append(line)

	reliable = dict()
	for path, lines in block_lines.items():
		hull = shapely.ops.cascaded_union([
			line.image_space_polygon for line in lines]).convex_hull
		reliable[path] = hull.intersection(
			all_blocks[path].image_space_polygon)

	return reliable


class LineExtractor:
	def __init__(self, line_height, options, min_confidence=0.5):
		self._options = options
		self._line_height = line_height
		assert self._line_height is not None

		if self._options["binarize"]:
			self._binarizer = Binarizer(
				self._options["binarize_window_size"])
		else:
			self._binarizer = None

		self._min_confidence = min_confidence

	def _extract_line_image(self, item):
		line_path, line, column = item

		return line_path, line.image(
			target_height=self._line_height,
			column=column,
			dewarped=not self._options["do_not_dewarp"],
			deskewed=not self._options["do_not_deskew"],
			binarizer=self._binarizer)

	def __call__(self, page_path, lines, ignored=[]):
		lines = dict(
			(k, v) for k, v in lines.items()
			if tuple(k[:2]) not in ignored)

		with open(page_path.with_suffix(".tables.json"), "r") as f:
			table_data = json.loads(f.read())
		columns = dict(
			(tuple(k.split("/")), xs)
			for k, xs in table_data["columns"].items())

		line_parts = []
		for path, line in lines.items():
			if line.confidence > self.min_confidence:
				line_columns = columns.get(path[:3])
				if line_columns is None:
					line_parts.append((path, line, None))
				else:
					line_columns = [None] + line_columns + [None]
					for i, (x0, x1) in enumerate(zip(line_columns, line_columns[1:])):
						line_parts.append((path + (str(i),), line, (x0, x1)))

		pool = multiprocessing.pool.ThreadPool(processes=8)
		return pool.map(self._extract_line_image, line_parts)
