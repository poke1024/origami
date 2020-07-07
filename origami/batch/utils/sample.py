#!/usr/bin/env python3

import click
import random
import logging
import shutil

from pathlib import Path
from tqdm import tqdm

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Annotation


class SampleProcessor(Processor):
	def __init__(self, options):
		options["nolock"] = True
		super().__init__(options)
		self._options = options

		self._paths = []
		self._out_path = Path(self._options["output_path"])
		self._out_path.mkdir(exist_ok=True)

	def artifacts(self):
		return [
			("data", Input(Annotation(self._options["annotation"])))
		]

	def should_process(self, p: Path) -> bool:
		return True

	def process(self, page_path: Path, data):
		self._paths.append(data.annotation)

	def output(self):
		k = self._options["number"]
		if k > len(self._paths):
			logging.error("not enough data to sample %d pages." % k)
			k = len(self._paths)
		paths = random.sample(self._paths, k)

		filename = self._options["filename"]
		for p in tqdm(paths, desc="copying"):
			if filename == "page":
				name = p.parent.with_suffix(p.suffix).name
			elif filename == "path":
				name = p.parent.with_suffix(p.suffix)
				sep = "--"
				name = str(name).replace("/", sep)
				name = name.strip(sep)
			else:
				raise ValueError(filename)
			shutil.copy(p, self._out_path / name)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-o', '--output-path',
	type=Path,
	required=True,
	help="Directory where samples will get saved.")
@click.option(
	'-n', '--number',
	type=int,
	default=10,
	help="Number of page results to sample.")
@click.option(
	'--annotation',
	type=str,
	default="layout",
	help="Type of annotation to sample.")
@click.option(
	'--filename',
	type=click.Choice(['page', 'path'], case_sensitive=False),
	default="page",
	help="How to name the sampled file.")
@Processor.options
def sample(data_path, **kwargs):
	""" Get a sample of page results in DATA_PATH. """
	processor = SampleProcessor(kwargs)
	processor.traverse(data_path)
	processor.output()


if __name__ == "__main__":
	sample()
