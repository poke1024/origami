#!/usr/bin/env python3

import click
import os
import collections

from pathlib import Path
from tabulate import tabulate

from origami.batch.core.processor import Processor
from origami.batch.core.io import Input, parse_artifact


class ResetProcessor(Processor):
	def __init__(self, artifact_spec, options):
		super().__init__(options)

		by_stage = collections.defaultdict(list)
		for s in artifact_spec.split(","):
			artifact, spec_stage = parse_artifact(s.strip())
			if spec_stage is None:
				if artifact.stages:
					stages = artifact.stages
				else:
					stages = [None]
			else:
				stages = [spec_stage]
			for stage in stages:
				by_stage[stage].append(artifact)
		self._artifacts = by_stage

		self._input_names = collections.defaultdict()
		for stage, artifacts in self._artifacts.items():
			stage_name = stage.name.lower() if stage else "all"
			self._input_names[stage] = "input_%s" % stage_name

	def print_artifacts(self):
		table = []
		for stage, artifacts in self._artifacts.items():
			for artifact in artifacts:
				table.append((
					artifact.name,
					stage.name.lower() if stage else "all"))

		print(tabulate(table, ["artifact", "stage"], tablefmt="github"))

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		inputs = []

		for stage, artifacts in self._artifacts.items():
			inputs.append((
				self._input_names[stage],
				Input(*artifacts, stage=stage, take_any=True)))

		return inputs

	def process(self, p: Path, **inputs):
		for stage, artifacts in self._artifacts.items():
			inp = inputs[self._input_names[stage]]
			for artifact in artifacts:
				p = inp.path(artifact)
				if p.exists():
					os.remove(p)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-a', '--artifacts',
	type=str,
	required=True,
	help="one or multiple artifacts, e.g. \"contours, order, lines, dewarping_transform, tables, compose\"")
@Processor.options
def reset(data_path, artifacts, **kwargs):
	""" Delete certain artifacts for all document images in DATA_PATH. """
	processor = ResetProcessor(artifacts, kwargs)

	print("The following artifacts will get deleted for all documents:")
	print("", flush=True)
	processor.print_artifacts()
	print("", flush=True)

	if click.confirm('Are you sure?'):
		processor.traverse(data_path)


if __name__ == "__main__":
	reset()

