import imghdr
import click
import zipfile
import json
import logging

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.block_processor import BlockProcessor
from origami.core.block import LineDetector


class LineDetectionProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	def should_process(self, p: Path) -> bool:
		return (imghdr.what(p) is not None) and\
			p.with_suffix(".contours.zip").exists() and\
			not p.with_suffix(".lines.zip").exists()

	def process(self, page_path: Path):
		blocks = self.read_blocks(page_path)

		lines_path = page_path.with_suffix(".lines.zip")
		with atomic_write(lines_path, mode="wb", overwrite=False) as f:

			with zipfile.ZipFile(f, "w", compression=self.compression) as zf:
				for parts, block in blocks.items():
					prediction_name = parts[0]
					class_name = parts[1]
					block_id = parts[2]

					try:
						detector = LineDetector(
							block,
							force_parallel_lines=self._options["force_parallel_lines"])

						lines = detector.detect_lines(
							fringe_limit=self._options["fringe_limit"],
							extra_height=self._options["extra_height"],
							text_buffer=self._options["text_buffer"])

						for line_id, line in enumerate(lines):
							line_name = "%s/%s/%s/%04d" % (
								prediction_name, class_name, block_id, line_id)
							zf.writestr("%s.json" % line_name, json.dumps(line.info))
					except:
						logging.error("failed to detect lines on block %s" % "/".join(parts))
						raise


@click.command()
@click.option(
	'-p', '--force-parallel-lines',
	default=False,
	is_flag=True,
	help='force parallel baselines inside a region')
@click.option(
	'-f', '--fringe-limit',
	default=0.1,
	type=float,
	help='ignore region fringes above this ratio')
@click.option(
	'-h', '--extra-height',
	default=0.05,
	type=float,
	help='compensate low Tesseract height estimation')
@click.option(
	'-b', '--text-buffer',
	default=15,
	type=int,
	help='text area boundary expansion in pixels')
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
def detect_lines(data_path, **kwargs):
	""" Perform line detection on all document images in DATA_PATH. Needs
	information from contours batch. """
	processor = LineDetectionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	detect_lines()

