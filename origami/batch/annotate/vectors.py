import imghdr
import click
import PIL.Image

from pathlib import Path
from PySide2 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.block_processor import BlockProcessor
from origami.batch.annotate.utils import render_blocks, render_separators


class DebugVectorsProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".contours.zip").exists()

	def process(self, p: Path):
		blocks = self.read_blocks(p)
		separators = self.read_separators(p)

		def get_label(block_path):
			classifier, segmentation_label, block_id = block_path
			return str(block_id)

		qt_im = ImageQt(PIL.Image.open(p))
		qt_im = render_blocks(qt_im, blocks, get_label)
		qt_im = render_separators(qt_im, separators)
		qt_im.save(str(p.with_suffix(".annotate.vectors.jpg")))


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
def debug_vectors(data_path, **kwargs):
	""" Export annotate information on vectors for all document images in DATA_PATH. """
	processor = DebugVectorsProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = QtGui.QGuiApplication()
	debug_vectors()
