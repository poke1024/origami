#!/usr/bin/env python3

import imghdr
import click
import zipfile
import json
import logging
import multiprocessing.pool

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.core.block import ConcurrentLineDetector


class WarpDetectionProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("input", Input(Artifact.CONTOURS, stage=Stage.WARPED)),
			("output", Output(Artifact.LINES, stage=Stage.WARPED))
		]

	def process(self, page_path: Path, input, output):
		detector = ConcurrentLineDetector(
			force_parallel_lines=False)

		block_lines = detector(input.blocks)

		with output.lines() as zf:
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
def detect_warp(data_path, **kwargs):
	""" Perform warp detection on all document images in DATA_PATH. Needs
	information from contours batch. """
	processor = WarpDetectionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	detect_warp()

