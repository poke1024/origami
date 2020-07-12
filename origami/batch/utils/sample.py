#!/usr/bin/env python3

import click
import random
import logging
import shutil
import enum
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


class NamingScheme(enum.Enum):
	PAGE = 0
	PATH = 1


def name_by_page(suffix, path):
	return path.parent.with_suffix(suffix).name


def name_by_path(suffix, path):
	name = path.parent.with_suffix(suffix)
	sep = "--"
	name = str(name).replace("/", sep)
	name = name.strip(sep)
	return name


_namers = {
	NamingScheme.PAGE: name_by_page,
	NamingScheme.PATH: name_by_path
}


def parse_artifact(name):
	if "/" in name:
		parts = list(map(
			lambda s: s.strip(), name.split("/")))
		if len(parts) != 2:
			raise ValueError(name)
		t, name = parts
		if t != "annotation":
			raise ValueError(name)
		artifact = Annotation(name)
	else:
		artifact = Artifact[name.upper()]

	return artifact


class SampleProcessor(Processor):
	def __init__(self, options):
		options["nolock"] = True
		super().__init__(options)
		self._options = options
		self._stage = Stage.ANY

		self._out_path = Path(self._options["output_path"])
		self._out_path.mkdir(exist_ok=True)

		self._namer = _namers[NamingScheme[
			self._options["filename"].upper()]]

		self._artifacts = []
		for spec in options["artifacts"].split(","):
			artifact = parse_artifact(spec.strip())

			if self._options["do_not_unpack"]:
				copy = _default_copy
			elif artifact == Artifact.COMPOSE:
				copy = _unpack_zip
			else:
				copy = _default_copy

			self._artifacts.append((artifact, copy))

		self._queue = []

	def artifacts(self):
		return [
			("data", Input(*[a for a, _ in self._artifacts], stage=self._stage))
		]

	def should_process(self, p: Path) -> bool:
		return True

	def process(self, page_path: Path, data):
		for artifact, copy in self._artifacts:
			copy_args = (artifact, data.path(artifact), copy)
			if self._options["all"]:  # take all?
				self._copy(*copy_args)
			else:
				self._queue.append(copy_args)

	def _copy(self, artifact, path, copy):
		copy(path, self._out_path / self._namer(
			"." + artifact.filename(self._stage), path))

	def output(self):
		if self._options["all"]:
			return

		k = self._options["number"]
		if k > len(self._queue):
			k = len(self._queue)
			logging.error("only found %d pages to sample from." % k)
		sampled = random.sample(self._queue, k)

		if not sampled:
			return

		for args in tqdm(sampled, desc="copying"):
			self._copy(*args)


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
	'-a', '--artifacts',
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
