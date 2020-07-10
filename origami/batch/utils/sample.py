#!/usr/bin/env python3

import click
import random
import logging
import shutil
import zipfile

from pathlib import Path
from tqdm import tqdm

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Annotation


def _default_copy(src, dst):
	shutil.copy(src, dst)


def _unpack_zip(src, dst):
	parent = dst.parent
	basename = dst.name.rsplit(".", 1)[0]

	with zipfile.ZipFile(src, "r") as zf:
		for name in zf.namelist():
			with open(parent / (basename + "_" + name), "wb") as f:
				f.write(zf.read(name))


class SampleProcessor(Processor):
	def __init__(self, options):
		options["nolock"] = True
		super().__init__(options)
		self._options = options

		self._paths = []
		self._out_path = Path(self._options["output_path"])
		self._out_path.mkdir(exist_ok=True)

		artifact = options["artifact"]
		if "/" in artifact:
			parts = map(lambda s: s.strip(), artifact.split("/"))
			if len(parts) != 2:
				raise ValueError(artifact)
			t, name = parts
			if t != "annotation":
				raise ValueError(artifact)
			self._artifact = Annotation(name)
		else:
			self._artifact = Artifact[artifact.upper()]

		if self._options["do_not_unpack"]:
			self._copy = _default_copy
		elif self._artifact == Artifact.COMPOSE:
			self._copy = _unpack_zip
		else:
			self._copy = _default_copy

	def artifacts(self):
		return [
			("data", Input(self._artifact))
		]

	def should_process(self, p: Path) -> bool:
		return True

	def process(self, page_path: Path, data):
		paths = data.paths
		assert len(paths) == 1
		self._paths.append(paths[0])

	def output(self):
		if self._options["all"]:  # take all?
			paths = self._paths
		else:
			k = self._options["number"]
			if k > len(self._paths):
				logging.error("not enough data to sample %d pages." % k)
				k = len(self._paths)
			paths = random.sample(self._paths, k)

		if paths:
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
				self._copy(p, self._out_path / name)


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
	'--all',
	is_flag=True,
	default=False,
	help="Ignore --n and take all.")
@click.option(
	'-a', '--artifact',
	type=str,
	default="annotation/layout",
	help="Artifact to sample.")
@click.option(
	'--do-not-unpack',
	is_flag=False,
	default=False,
	help="Do not unpack zip files.")
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
