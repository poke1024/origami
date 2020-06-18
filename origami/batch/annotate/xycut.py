import imghdr
import click
import PIL.Image
import json
import math
import cv2
import numpy as np

from pathlib import Path
from PySide2 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.block_processor import BlockProcessor
from origami.batch.annotate.utils import render_blocks


class DebugXYCutProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".contours.zip").exists() and\
			p.with_suffix(".xycut.json").exists()

	def process(self, p: Path):
		blocks = self.read_blocks(p)
		with open(p.with_suffix(".xycut.json"), "r") as f:
			xycut_data = json.loads(f.read())

		order = dict((tuple(path.split("/")), i) for i, path in enumerate(xycut_data["order"]))
		blocks = dict((path, blocks[path]) for path in blocks.keys() if path in order)

		def get_label(block_path):
			return str(1 + order[block_path])

		matrix = cv2.getRotationMatrix2D(
			(0, 0), xycut_data["skew"] * (180 / math.pi), 1)
		im = PIL.Image.open(p)
		pixels = cv2.warpAffine(np.array(im), matrix, (im.width, im.height))
		im = PIL.Image.fromarray(pixels)

		qt_im = ImageQt(im)
		qt_im = render_blocks(qt_im, blocks, get_label, matrix=matrix)
		qt_im.save(str(p.with_suffix(".annotate.xycut.jpg")))


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
def debug_xycut(data_path, **kwargs):
	""" Export annotate information on xycuts for all document images in DATA_PATH. """
	processor = DebugXYCutProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = QtGui.QGuiApplication()
	debug_xycut()
