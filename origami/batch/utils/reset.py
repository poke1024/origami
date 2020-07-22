#!/usr/bin/env python3

import click
import os

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output, parse_artifact


class ResetProcessor(Processor):
	def __init__(self, artifact, options):
		super().__init__(options)
		self._artifact = parse_artifact(artifact)

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("input", Input(self._artifact)),
		]

	def process(self, p: Path, input):
		os.remove(input.path(self._artifact))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-a', '--artifact',
	type=str,
	required=True)
@Processor.options
def reset(data_path, artifact, **kwargs):
	""" Delete certain artifacts for all document images in DATA_PATH. """
	processor = ResetProcessor(artifact, kwargs)

	if click.confirm('Delete all %s artifacts?' % artifact):
		processor.traverse(data_path)


if __name__ == "__main__":
	reset()

