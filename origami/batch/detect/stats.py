#!/usr/bin/env python3

import imghdr
import click
import collections

from pathlib import Path
from tabulate import tabulate

from origami.batch.core.processor import Processor
from origami.batch.core.io import find_data_path


class StatsProcessor(Processor):
	def __init__(self, options):
		options["nolock"] = True
		super().__init__(options)

		self._list_names = options["list"]
		if self._list_names:
			self._names = []
		else:
			self._names = None

		self._num_pages = 0
		self._artifacts = collections.defaultdict(int)

	def artifacts(self):
		return []

	def should_process(self, p: Path) -> bool:
		return True

	def process(self, page_path: Path):
		if self._list_names:
			self._names.append(page_path.name)

		self._num_pages += 1

		data_path = find_data_path(page_path)
		for p in data_path.iterdir():
			if not p.name.startswith("."):
				self._artifacts[p.name] += 1

	def print(self):
		data = []

		data.append(["pages", str(self._num_pages)])
		for name, n in sorted(self._artifacts.items()):
			data.append([name, str(n)])

		print(tabulate(data, tablefmt="psql"))

		if self._list_names:
			for name in self._names:
				print(name)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-l', '--list',
	is_flag=True,
	default=False,
	help="List found page names.")
@click.option(
	'--name',
	type=str,
	default="",
	help="Only process paths that conform to the given pattern.")
def stats(data_path, **kwargs):
	""" List stats of pages in DATA_PATH. """
	processor = StatsProcessor(kwargs)
	processor.traverse(data_path)
	processor.print()


if __name__ == "__main__":
	stats()
