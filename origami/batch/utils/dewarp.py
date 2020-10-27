#!/usr/bin/env python3

import click

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output, Annotation


class DewarpProcessor(Processor):
	def __init__(self, options):
		super().__init__(options, needs_qt=True)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("dewarped", Input(
				Artifact.DEWARPING_TRANSFORM, stage=Stage.DEWARPED)),
			("output", Output(Annotation("dewarped")))
		]

	def process(self, page_path: Path, dewarped, output):
		output.annotation(dewarped.page.dewarped)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@Processor.options
def dewarp(data_path, **kwargs):
	""" Create dewarped images for all document images in DATA_PATH. """
	processor = DewarpProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	dewarp()
