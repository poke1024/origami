import imghdr
import click
import PIL.Image

from pathlib import Path
from PySide2 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.block_processor import Processor
from origami.batch.annotate.utils import render_lines


class DebugWarpProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".warped.lines.zip").exists()

	def process(self, page_path: Path):
		blocks = self.read_blocks(page_path)
		lines = self.read_lines(page_path, blocks)

		def get_label(lines_path):
			classifier, segmentation_label, block_id, line_id = lines_path
			return (classifier, segmentation_label), line_id

		qt_im = ImageQt(PIL.Image.open(page_path))
		pixmap = QtGui.QPixmap.fromImage(qt_im)
		pixmap = render_lines(pixmap, lines, get_label)
		pixmap.toImage().save(str(page_path.with_suffix(".annotate.warp.jpg")))


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
def debug_warp(data_path, **kwargs):
	""" Export annotate information on lines for all document images in DATA_PATH. """
	processor = DebugWarpProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = QtGui.QGuiApplication()
	debug_warp()
