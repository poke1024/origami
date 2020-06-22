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
from origami.core.page import Page
from origami.batch.annotate.utils import render_blocks


class DebugXYCutProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".dewarped.contours.zip").exists() and\
			p.with_suffix(".dewarped.transform.zip").exists() and\
			p.with_suffix(".xycut.json").exists() and\
			not p.with_suffix(".annotate.xycut.jpg").exists()

	def process(self, p: Path):
		blocks = self.read_dewarped_blocks(p)

		with open(p.with_suffix(".xycut.json"), "r") as f:
			xycut_data = json.loads(f.read())

		order = dict((tuple(path.split("/")), i) for i, path in enumerate(xycut_data["order"]))
		blocks = dict((path, blocks[path]) for path in blocks.keys() if path in order)

		def get_label(block_path):
			return str(1 + order[block_path])

		page = Page(p, dewarp=True)

		qt_im = ImageQt(page.dewarped)
		pixmap = QtGui.QPixmap.fromImage(qt_im)
		pixmap = render_blocks(pixmap, blocks, get_label)
		pixmap.toImage().save(str(p.with_suffix(".annotate.xycut.jpg")))


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
