import imghdr
import click
import numpy as np
import wquantiles
import cv2
import math
import json
import shapely.affinity

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.block_processor import BlockProcessor
from origami.core.math import to_shapely_matrix
from origami.core.xycut import reading_order


class XYCutProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".lines.zip").exists() and\
			not p.with_suffix(".xycut.json").exists()

	def process(self, page_path: Path):
		blocks = self.read_blocks(page_path)
		lines = self.read_lines(page_path, blocks)

		if len(lines) < 1:
			return

		angles = np.array([line.angle for line in lines.values()])
		lengths = np.array([line.length for line in lines.values()])
		skew = wquantiles.median(angles, lengths)

		m = to_shapely_matrix(cv2.getRotationMatrix2D(
			(0, 0), skew * (180 / math.pi), 1))

		names = []
		bounds = []
		for block_path, block in blocks.items():
			names.append("/".join(block_path))
			bounds.append(shapely.affinity.affine_transform(
				block.image_space_polygon, m).bounds)

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
