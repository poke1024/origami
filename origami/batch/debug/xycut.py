import imghdr
import click
import PIL.Image
import json

from pathlib import Path
from PySide2 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.block_processor import BlockProcessor
from origami.batch.debug.utils import render_blocks


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

		def get_label(block_path):
			return str(1 + xycut_data["order"].index("/".join(block_path)))

		qt_im = ImageQt(PIL.Image.open(p))
		qt_im = render_blocks(qt_im, blocks, get_label)
		qt_im.save(str(p.with_suffix(".debug.xycut.jpg")))


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
	""" Export debug information on xycuts for all document images in DATA_PATH. """
	processor = DebugXYCutProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = QtGui.QGuiApplication()
	debug_xycut()
