import imghdr
import click
import PIL.Image
import io
import numpy as np

from pathlib import Path
from PySide2 import QtGui, QtCore
from PIL.ImageQt import ImageQt

from origami.batch.core.block_processor import BlockProcessor
from origami.batch.annotate.utils import render_warped_line_paths


class DebugLinesProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".binarized.png").exists() and\
			p.with_suffix(".dewarped.lines.zip").exists()

	def process(self, page_path: Path):
		blocks = self.read_dewarped_blocks(page_path)
		lines = self.read_dewarped_lines(page_path, blocks)

		if not blocks:
			return

		page = list(blocks.values())[0].page
		predictors = self.read_predictors(page_path)

		qt_im = ImageQt(page.warped)
		pixmap = QtGui.QPixmap.fromImage(qt_im)
		pixmap = render_warped_line_paths(pixmap, lines, predictors)

		buffer = QtCore.QBuffer()
		buffer.open(QtCore.QBuffer.ReadWrite)
		pixmap.toImage().save(buffer, "PNG")
		im = PIL.Image.open(io.BytesIO(buffer.data())).convert("RGB")

		bin = PIL.Image.open(page_path.with_suffix(".binarized.png"))
		alpha = np.array(bin.convert("L")) // 2
		im = PIL.Image.composite(im, bin.convert("RGB"), PIL.Image.fromarray(alpha))

		im.save(page_path.with_suffix(".annotate.lines.jpg"))



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
def debug_lines(data_path, **kwargs):
	""" Export annotate information on lines for all document images in DATA_PATH. """
	processor = DebugLinesProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = QtGui.QGuiApplication()
	debug_lines()
