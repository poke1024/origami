import imghdr
import click
import PIL.Image
import json
import math
import cv2
import collections
import shapely.ops
import numpy as np

from pathlib import Path
from PySide2 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.block_processor import BlockProcessor
from origami.core.page import Page
from origami.batch.annotate.utils import render_contours


def reliable_contours(all_blocks, all_lines, min_confidence=0.5):
	block_lines = collections.defaultdict(list)
	for path, line in all_lines.items():
		if line.confidence > min_confidence:
			block_lines[path[:3]].append(line)

	result = dict()
	for path, lines in block_lines.items():
		hull = shapely.ops.cascaded_union([
			line.image_space_polygon for line in lines]).convex_hull
		result[path] = hull.intersection(
			all_blocks[path].image_space_polygon)

	return result


class DebugLayoutProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".aggregate.contours.zip").exists() and\
			p.with_suffix(".dewarped.transform.zip").exists() and\
			p.with_suffix(".xycut.json").exists() and\
			not p.with_suffix(".annotate.layout.jpg").exists()

	def process(self, page_path: Path):
		blocks = self.read_aggregate_blocks(page_path)
		lines = self.read_aggregate_lines(page_path, blocks)

		with open(page_path.with_suffix(".xycut.json"), "r") as f:
			xycut_data = json.loads(f.read())

		order = dict(
			(tuple(path.split("/")), i)
			for i, path in enumerate(xycut_data["order"]["*"]))
		blocks = dict(
			(path, blocks[path])
			for path in blocks.keys() if path in order)

		def get_label(block_path):
			return block_path[:2], str(1 + order[block_path])

		page = Page(page_path, dewarp=True)
		predictors = self.read_predictors(page_path)

		qt_im = ImageQt(page.dewarped)
		pixmap = QtGui.QPixmap.fromImage(qt_im)

		contours = reliable_contours(blocks, lines)
		pixmap = render_contours(pixmap, contours, get_label, predictors)

		pixmap.toImage().save(str(
			page_path.with_suffix(".annotate.layout.jpg")))


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
def debug_layout(data_path, **kwargs):
	""" Export annotate information on xycuts for all document images in DATA_PATH. """
	processor = DebugLayoutProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = QtGui.QGuiApplication()
	debug_layout()
