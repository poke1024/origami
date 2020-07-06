#!/usr/bin/env python3

import imghdr
import click
import shutil

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output

from origami.core.segment import Segmentation


class SegmentationConverter(Processor):
	def __init__(self, options):
		super().__init__(options)

	def should_process(self, p):
		return (
			p.with_suffix(".segment.zip").exists() or
			p.with_suffix(".sgm.pickle").exists())

	def artifacts(self):
		return [
			("output", Output(Artifact.SEGMENTATION))
		]

	def process(self, p: Path, output):
		# util old segmentation filename.
		old_zip_path = p.with_suffix(".segment.zip")
		if old_zip_path.exists():
			shutil.move(old_zip_path, output.paths[0])
		else:
			# util very old segmentation pickle file format.
			pickle_path = p.with_suffix(".sgm.pickle")
			if pickle_path.exists():
				segmentation = Segmentation.open_pickle(pickle_path)
				segmentation.save(output.paths[0])


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
def segment(data_path, **kwargs):
	""" Convert page segmentation data from old to new format. """
	processor = SegmentationConverter(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	segment()

