import imghdr
import click
import PIL.Image
import skimage.filters
import skimage.morphology
import numpy as np

from pathlib import Path

from origami.batch.core.processor import Processor


class BinarizationProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			not p.with_suffix(".binarized.png").exists()

	def process(self, p: Path):
		im = PIL.Image.open(p)

		pixels = np.array(im.convert("L"))
		thresh_sauvola = skimage.filters.threshold_sauvola(
			pixels, window_size=self._options["window_size"])
		binarized = pixels > thresh_sauvola

		if self._options["despeckle"] > 0:
			binarized = np.logical_not(skimage.morphology.remove_small_objects(
				np.logical_not(binarized), min_size=self._options["despeckle"]))

		PIL.Image.fromarray(binarized.astype(np.uint8) * 0xff, "L").save(
			p.with_suffix(".binarized.png"))


@click.command()
@click.option(
	'-w', '--window-size',
	type=int,
	default=15,
	help='window size for binarization')
@click.option(
	'-d', '--despeckle',
	type=int,
	default=40,
	help='despeckle blotches up to this area')
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
def binarize(data_path, **kwargs):
	""" Perform binarization on all document images in DATA_PATH. """
	processor = BinarizationProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	binarize()
