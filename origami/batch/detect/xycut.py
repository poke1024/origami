import imghdr
import click
import numpy as np
import wquantiles
import cv2
import math
import json
import shapely.affinity
import collections

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.block_processor import BlockProcessor
from origami.core.math import to_shapely_matrix
from origami.core.xycut import reading_order


class XYCutProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._buffer = 10
		self._block_split_limit = 0

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".lines.zip").exists() and\
			not p.with_suffix(".xycut.json").exists()

	def process(self, page_path: Path):
		blocks = self.read_blocks(page_path)
		lines = self.read_lines(page_path, blocks)

		if len(lines) < 1:
			return

		lines_by_block = collections.defaultdict(list)
		for line_path, line in lines.items():
			lines_by_block[line_path[:3]].append((line_path, line))
		for k in list(lines_by_block.keys()):
			lines_by_block[k] = sorted(lines_by_block[k], key=lambda x: x[0])

		angles = np.array([line.angle for line in lines.values()])
		lengths = np.array([line.length for line in lines.values()])
		skew = wquantiles.median(angles, lengths)

		m = to_shapely_matrix(cv2.getRotationMatrix2D(
			(0, 0), skew * (180 / math.pi), 1))

		names = []
		bounds = []

		def add(polygon, path):
			minx, miny, maxx, maxy = shapely.affinity.affine_transform(polygon, m).bounds

			minx = min(minx + self._buffer, maxx)
			maxx = max(maxx - self._buffer, minx)
			miny = min(miny + self._buffer, maxy)
			maxy = max(maxy - self._buffer, miny)

			bounds.append((minx, miny, maxx, maxy))
			names.append("/".join(path))

		for block_path, block in blocks.items():
			block_lines = lines_by_block[block_path]
			if len(block_lines) < 1:
				continue
			elif 1 < len(block_lines) <= self._block_split_limit:
				for line_path, line in block_lines:
					add(line.image_space_polygon, line_path)
			else:
				add(block.image_space_polygon, block_path)

		data = dict(
			skew=skew,
			order=[names[i] for i in reading_order(bounds)])

		zf_path = page_path.with_suffix(".xycut.json")
		with atomic_write(zf_path, mode="w", overwrite=False) as f:
			f.write(json.dumps(data))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--nolock',
	is_flag=True,
	default=False,
	help="Do not lock files while processing. Breaks concurrent batches, "
	"but is necessary on some network file systems.")
def xy_cut(data_path, **kwargs):
	""" Perform XYCut on all pages in DATA_PATH. """
	processor = XYCutProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	xy_cut()
