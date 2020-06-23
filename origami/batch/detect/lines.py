import imghdr
import click
import zipfile
import json
import logging
import multiprocessing.pool

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.block_processor import BlockProcessor
from origami.core.block import ConcurrentLineDetector


class LineDetectionProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return (imghdr.what(p) is not None) and\
			p.with_suffix(".dewarped.contours.zip").exists() and\
			p.with_suffix(".dewarped.transform.zip").exists() and\
			not p.with_suffix(".dewarped.lines.zip").exists()

	def process(self, page_path: Path):
		blocks = self.read_dewarped_blocks(page_path)

		detector = ConcurrentLineDetector(
			force_parallel_lines=False,
			extra_height=self._options["extra_height"],
			extra_descent=self._options["extra_descent"],
			contours_buffer=self._options["contours_buffer"],
			contours_concavity=self._options["contours_concavity"],
			contours_detail=self._options["contours_detail"])

		block_lines = detector(blocks)

		lines_path = page_path.with_suffix(".dewarped.lines.zip")
		with atomic_write(lines_path, mode="wb", overwrite=False) as f:

			with zipfile.ZipFile(f, "w", compression=self.compression) as zf:
				info = dict(version=1)
				zf.writestr("meta.json", json.dumps(info))

				for parts, lines in block_lines.items():
					prediction_name = parts[0]
					class_name = parts[1]
					block_id = parts[2]

					for line_id, line in enumerate(lines):
						line_name = "%s/%s/%s/%04d" % (
							prediction_name, class_name, block_id, line_id)
						zf.writestr("%s.json" % line_name, json.dumps(line.info))


@click.command()
@click.option(
	'--extra-height',
	default=0.075,
	type=float,
	help='compensate low Tesseract height estimation')
@click.option(
	'--extra-descent',
	default=0.025,
	type=float,
	help='compensate low Tesseract descent estimation')
@click.option(
	'--contours-buffer',
	default=0.0015,
	type=float,
	help='contour boundary expansion')
@click.option(
	'--contours-concavity',
	default=8,
	type=float,
	help='contour concavity')
@click.option(
	'--contours-detail',
	default=0.01,
	type=float,
	help='contour detail in terms of segment length')
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--name',
	type=str,
	default="",
	help="Only process paths that conform to the given pattern.")
@click.option(
	'--nolock',
	is_flag=True,
	default=False,
	help="Do not lock files while processing. Breaks concurrent batches, "
	"but is necessary on some network file systems.")
def detect_lines(data_path, **kwargs):
	""" Perform line detection on all document images in DATA_PATH. Needs
	information from contours batch. """
	processor = LineDetectionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	detect_lines()

