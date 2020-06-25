import imghdr
import click

from pathlib import Path

from origami.batch.core.processor import Processor

from origami.core.segment import Segmentation


class SegmentationConverter(Processor):
	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			(p.with_suffix(".sgm.pickle")).exists() and\
			not p.with_suffix(".segment.zip").exists()

	def process(self, p: Path):
		segmentation = Segmentation.open_pickle(p.with_suffix(".sgm.pickle"))
		segmentation.save(p.with_suffix(".segment.zip"))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
def segment(data_path):
	""" Convert page segmentation data from old to new format. """
	processor = SegmentationConverter()
	processor.traverse(data_path)


if __name__ == "__main__":
	segment()

