import imghdr
import click
import PIL.Image

from pathlib import Path
from PySide2 import QtGui

from origami.batch.core.block_processor import BlockProcessor
from origami.batch.debug.utils import render_blocks


class DebugContoursProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".contours.zip").exists()

	def process(self, p: Path):
		blocks = self.read_blocks(p)

		def get_label(block_path):
			classifier, segmentation_label, block_id = block_path
			return str(block_id)

		im = render_blocks(PIL.Image.open(p), blocks, get_label)
		im.save(str(p.with_suffix(".debug.contours.jpg")))


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
def debug_contours(data_path, **kwargs):
	""" Export debug information on contours for all document images in DATA_PATH. """
	processor = DebugContoursProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = QtGui.QGuiApplication()
	debug_contours()
