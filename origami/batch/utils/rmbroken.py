#!/usr/bin/env python3

import click
import json
import os
import time

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Stage, Input


class RemoveBrokenProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._stale = 60 * 60

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("reader", Input(stage=Stage.ANY))
		]

	def process(self, p: Path, reader):
		obsolete = []

		for p in reader.data_path.iterdir():
			if p.name.endswith(".json"):
				ok = True
				with open(p, "r") as f:
					try:
						json.loads(f.read())
					except json.decoder.JSONDecodeError:
						ok = False
				if not ok:
					obsolete.append(p)

			if p.name.startswith("tmp"):
				if os.path.getmtime(p) - time.time() > self._stale:
					obsolete.append(p)

		for p in obsolete:
			os.remove(p)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@Processor.options
def rmbroken(data_path, **kwargs):
	""" Remove broken files in DATA_PATH. """
	processor = RemoveBrokenProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	rmbroken()
