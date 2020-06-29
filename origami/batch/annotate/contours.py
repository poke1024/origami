import imghdr
import click
import PIL.Image
import logging

from pathlib import Path
from PySide2 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.block_processor import BlockProcessor
from origami.batch.annotate.utils import render_blocks, render_separators


class DebugContoursProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".warped.contours.zip").exists()

	def process(self, page_path: Path):
		blocks = self.read_blocks(page_path)
		separators = self.read_separators(page_path)

		if not blocks:
			logging.info("no blocks for %s" % page_path)
			return

		page = list(blocks.values())[0].page
		predictors = self.read_predictors(page_path)

		qt_im = ImageQt(page.warped)
		pixmap = QtGui.QPixmap.fromImage(qt_im)

		def get_label(block_path):
			classifier, segmentation_label, block_id = block_path
			return (classifier, segmentation_label), int(block_id)

		pixmap = render_blocks(pixmap, blocks, get_label, predictors)
		pixmap = render_separators(pixmap, separators)

		pixmap.toImage().save(str(
			page_path.with_suffix(".annotate.warped.contours.jpg")))


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
	""" Export annotate information on vectors for all document images in DATA_PATH. """
	processor = DebugContoursProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = QtGui.QGuiApplication()
	debug_contours()
