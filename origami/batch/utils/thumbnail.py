#!/usr/bin/env python3

import click

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output


class ThumbnailProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._size = options["size"]
		self._quality = options["quality"]

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("input", Input(stage=Stage.WARPED)),
			("output", Output(Artifact.THUMBNAIL))
		]

	def process(self, p: Path, input, output):
		im = input.page.warped
		im = im.convert("L")
		im.thumbnail((self._size, self._size))
		im.save(
			output.path(Artifact.THUMBNAIL),
			"JPEG", quality=self._quality, optimize=True)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--size',
	type=int,
	default=512)
@click.option(
	'--quality',
	type=int,
	default=30)
@Processor.options
def thumbnails(data_path, **kwargs):
	""" Compute thumbnails for all document images in DATA_PATH. """
	processor = ThumbnailProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	thumbnails()
