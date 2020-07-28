#!/usr/bin/env python3

import click
import json
import numpy as np
import cv2

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.core.block import ConcurrentLineDetector, TextAreaFactory
from origami.batch.core.utils import RegionsFilter


def scale_grid(s0, s1, grid):
	h0, w0 = s0
	h1, w1 = s1
	grid[:, :, 0] *= w1 / w0
	grid[:, :, 1] *= h1 / h0


class ConfidenceSampler:
	def __init__(self, blocks, segmentation):
		self._predictions = dict()
		for p in segmentation.predictions:
			self._predictions[p.name] = p

		self._page = list(blocks.values())[0].page
		self._page_shape = tuple(reversed(self._page.warped.size))

	def __call__(self, path, line, res=0.5):
		prediction_name, predictor_class = path[:2]

		predictor = self._predictions[prediction_name]
		lineclass = predictor.classes[predictor_class]

		grid = line.warped_grid(xres=res, yres=res)

		scale_grid(self._page_shape, predictor.labels.shape, grid)
		labels = cv2.remap(predictor.labels, grid, None, cv2.INTER_NEAREST)

		counts = np.bincount(labels.flatten(), minlength=len(predictor.classes))
		counts[predictor.classes["BACKGROUND"].value] = 0
		sum_all = np.sum(counts)
		if sum_all < 1:
			return 0
		return counts[lineclass.value] / sum_all


class LineDetectionProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._allow_conflicts = RegionsFilter(options["allow_conflicts"])
		self._min_confidence = options["min_confidence"]

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("warped", Input(Artifact.SEGMENTATION, stage=Stage.WARPED)),
			("aggregate", Input(Artifact.CONTOURS, stage=Stage.AGGREGATE)),
			("output", Output(Artifact.LINES, stage=Stage.AGGREGATE))
		]

	def process(self, page_path: Path, warped, aggregate, output):
		blocks = aggregate.regions.by_path
		if not blocks:
			return

		sampler = ConfidenceSampler(blocks, warped.segmentation)

		conflicting_blocks = [
			block for path, block in blocks.items()
			if not self._allow_conflicts(path)]

		detector = ConcurrentLineDetector(
			text_area_factory=TextAreaFactory(
				conflicting_blocks,
				buffer=self._options["contours_buffer"]),
			force_parallel_lines=False,
			force_lines=True,
			extra_height=self._options["extra_height"],
			extra_descent=self._options["extra_descent"])

		block_lines = detector(blocks)

		for block_path, lines in block_lines.items():
			for line in lines:
				line.update_confidence(sampler(block_path, line))

		with output.lines() as zf:
			info = dict(version=1, min_confidence=self._min_confidence)
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
	help='compensate underestimated line height')
@click.option(
	'--extra-descent',
	default=0.025,
	type=float,
	help='compensate underestimated line descent')
@click.option(
	'--contours-buffer',
	default=0.001,
	type=float,
	help='expand contours by specified relative amount')
@click.option(
	'--allow-conflicts',
	default="regions/ILLUSTRATION",
	type=str,
	help='regions types that may overlap without being resolved')
@click.option(
	'--min-confidence',
	type=float,
	default=0)
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@Processor.options
def detect_lines(data_path, **kwargs):
	""" Perform line detection on all document images in DATA_PATH. Needs
	information from contours batch. """
	processor = LineDetectionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	detect_lines()

