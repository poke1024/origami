import imghdr
import click
import zipfile
import json
import logging
import multiprocessing.pool
import numpy as np
import cv2

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.block_processor import BlockProcessor
from origami.core.block import ConcurrentLineDetector, TextAreaFactory
from origami.core.segment import Segmentation
from origami.core.predict import PredictorType


def scale_grid(s0, s1, grid):
	h0, w0 = s0
	h1, w1 = s1
	grid[:, :, 0] *= w1 / w0
	grid[:, :, 1] *= h1 / h0


class ConfidenceSampler:
	def __init__(self, page_path, blocks):
		segmentation = Segmentation.open(
			page_path.with_suffix(".segment.zip"))
		self._predictions = dict()
		for p in segmentation.predictions:
			self._predictions[p.name] = p

		self._page = list(blocks.values())[0].page
		self._page_shape = tuple(reversed(self._page.warped.size))

	def __call__(self, path, line, res=0.5):
		prediction_name, predictor_class = path[:2]

		predictor = self._predictions[prediction_name]
		lineclass = predictor.classes[predictor_class]

		grid = line.dewarped_grid(xres=res, yres=res)

		scale_grid(self._page_shape, predictor.labels.shape, grid)
		labels = cv2.remap(predictor.labels, grid, None, cv2.INTER_NEAREST)

		counts = np.bincount(labels.flatten(), minlength=len(predictor.classes))
		counts[predictor.classes["BACKGROUND"].value] = 0
		sum_all = np.sum(counts)
		if sum_all < 1:
			return 0
		return counts[lineclass.value] / sum_all


class LineDetectionProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._overwrite = self._options["overwrite"]

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return (imghdr.what(p) is not None) and\
			p.with_suffix(".segment.zip").exists() and\
			p.with_suffix(".aggregate.contours.zip").exists() and(
				self._overwrite or
				not p.with_suffix(".aggregate.lines.zip").exists())

	def process(self, page_path: Path):
		blocks = self.read_aggregate_blocks(page_path)
		if not blocks:
			return

		sampler = ConfidenceSampler(page_path, blocks)

		detector = ConcurrentLineDetector(
			text_area_factory=TextAreaFactory(
				list(blocks.values()),
				buffer=self._options["contours_buffer"]),
			force_parallel_lines=False,
			force_lines=True,
			extra_height=self._options["extra_height"],
			extra_descent=self._options["extra_descent"])

		block_lines = detector(blocks)

		for block_path, lines in block_lines.items():
			for line in lines:
				line.update_confidence(sampler(block_path, line))

		lines_path = page_path.with_suffix(".aggregate.lines.zip")
		with atomic_write(lines_path, mode="wb", overwrite=self._overwrite) as f:

			with zipfile.ZipFile(f, "w", compression=self.compression) as zf:
				info = dict(version=1)
				zf.writestr("meta.json", json.dumps(info))

				for parts, lines in block_lines.items():
					prediction_name = parts[0]
					class_name = parts[1]
					block_id = parts[2]

					for line_id, line in enumerate(lines):
						line_name = "%s/%s/%s/%d" % (
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
	help='contour expansion')
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
@click.option(
	'--overwrite',
	is_flag=True,
	default=False,
	help="Recompute and overwrite existing result files.")
def detect_lines(data_path, **kwargs):
	""" Perform line detection on all document images in DATA_PATH. Needs
	information from contours batch. """
	processor = LineDetectionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	detect_lines()

