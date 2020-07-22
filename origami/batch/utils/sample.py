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
from origami.batch.core.io import Artifact, Stage, Input, Annotation, parse_artifact


class DirectoryTarget:
	def __init__(self, dst):
		self._dst = dst
		self._dst.mkdir(exist_ok=True)

	def close(self):
		pass

	def default_copy(self, src, name):
		shutil.copy(src, self._dst / name)

	def unpack_zip(self, src, name):
		basename = name.rsplit(".", 1)[0]

		with zipfile.ZipFile(src, "r") as zf:
			for name in zf.namelist():
				with open(self._dst / (basename + "_" + name), "wb") as f:
					f.write(zf.read(name))


class ZipFileTarget:
	def __init__(self, dst):
		self._dst = dst
		self._zf = None
		self._closed = False

	def close(self):
		self._closed = True
		if self._zf:
			self._zf.close()

	@property
	def zf(self):
		if self._closed:
			raise RuntimeError("file already closed.")
		if self._zf is None:
			self._zf = zipfile.ZipFile(
				self._dst, "w", compression=zipfile.ZIP_DEFLATED)
		return self._zf

	def default_copy(self, src, name):
		with open(src, "rb") as f:
			self.zf.writestr(name, f.read())

	def unpack_zip(self, src, name):
		basename = name.rsplit(".", 1)[0]

		with zipfile.ZipFile(src, "r") as zf:
			for name in zf.namelist():
				self.zf.write(basename + "_" + name, zf.read(name))


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


class SampleProcessor(Processor):
	def __init__(self, options):
		options["nolock"] = True
		super().__init__(options)
		self._options = options
		self._stage = Stage.ANY

		self._out_path = Path(self._options["output_path"])
		if self._out_path.suffix == ".zip":
			self._target = ZipFileTarget(self._out_path)
		else:
			self._target = DirectoryTarget(self._out_path)

		self._namer = _namers[NamingScheme[
			self._options["filename"].upper()]]

		self._artifacts = []
		for spec in options["artifacts"].split(","):
			artifact = parse_artifact(spec.strip())

			if self._options["do_not_unpack"]:
				copy = self._target.default_copy
			elif artifact == Artifact.COMPOSE:
				copy = self._target.unpack_zip
			else:
				copy = self._target.default_copy

			self._artifacts.append((artifact, copy))

		self._queue = []

	def close(self):
		self._target.close()

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
		copy(path, self._namer(
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
	help="Directory or zip file where samples will get saved.")
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
	try:
		processor.traverse(data_path)
		processor.output()
	finally:
		processor.close()


if __name__ == "__main__":
	sample()
