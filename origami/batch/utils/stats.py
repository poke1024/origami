#!/usr/bin/env python3

import click
import json
import collections
import numpy as np
import pandas as pd
import logging

from pathlib import Path
from tabulate import tabulate

from origami.batch.core.processor import Processor
from origami.batch.core.io import find_data_path


def traceback_dict():
	return collections.defaultdict(list)


class StatsProcessor(Processor):
	def __init__(self, options):
		options["lock_strategy"] = "NONE"
		options["processes"] = 1
		super().__init__(options)

		self._list_names = options["list_names"]
		if self._list_names:
			self._names = []
		else:
			self._names = None

		self._list_errors = options["list_errors"]
		self._tracebacks = collections.defaultdict(traceback_dict)

		self._num_pages = 0
		self._artifacts = collections.defaultdict(int)
		self._times = collections.defaultdict(list)

	def parse_runtime_data(self, page_path, path):
		with open(path, "r") as f:
			try:
				runtime_data = json.loads(f.read())
			except json.decoder.JSONDecodeError as e:
				logging.warning("bad json at %s: %s" % (path, e))
				return

			for batch, data in runtime_data.items():
				t = data.get("elapsed")
				if t is None:
					t = data.get("total_time")  # legacy
				if t is not None:
					self._times[batch].append(t)

				if self._list_errors:
					if data.get("status") == "FAILED":
						self._tracebacks[batch][data.get("traceback")].append(page_path)

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
			if p.name.startswith(".") or p.name.startswith("tmp"):
				continue

			self._artifacts[p.name] += 1

			if p.name == "runtime.json":
				self.parse_runtime_data(page_path, p)

	def print_artifacts(self):
		data = []
		data.append(["pages", str(self._num_pages)])
		for name, n in sorted(self._artifacts.items()):
			data.append([name, str(n)])
		print(tabulate(data, tablefmt="psql"))

	def print_elapsed(self):
		data = []
		# as max times will be distorted by long setup
		# times for GPU related stuff, we actually use
		# a quantile here.
		for k in sorted(list(self._times.keys())):
			v = self._times[k]
			data.append((
				k,
				"%.1f" % np.min(v),
				"%.1f" % np.median(v),
				"%.1f" % np.quantile(v, 0.9)))
		print(tabulate(
			data,
			tablefmt="psql",
			headers=["batch", "min", "median", "max"]))

	def print_errors(self):
		full_data = dict(
			frequency=[],
			batch=[],
			traceback=[],
			pages=[])

		data = []
		for batch in sorted(list(self._tracebacks.keys())):
			tracebacks = self._tracebacks[batch]
			for k in sorted(list(tracebacks.keys())):
				paths = tracebacks[k]
				pages = "%d: %s" % (len(paths), paths[0].name)
				if len(paths) > 1:
					pages += ", ..."
				data.append((batch, k[-30:], pages))

				full_data["frequency"].append(len(paths))
				full_data["batch"].append(batch)
				full_data["traceback"].append(k)
				full_data["pages"].append(", ".join(map(str, paths)))

		print(tabulate(
			data,
			tablefmt="psql",
			headers=["batch", "traceback", "pages"]))

		df = pd.DataFrame.from_dict(full_data)
		with pd.ExcelWriter('errors.xlsx') as writer:
			df.to_excel(writer)

	def print(self):
		if self._artifacts:
			print("artifacts.")
			self.print_artifacts()

		if self._times:
			print("")
			print("elapsed.")
			self.print_elapsed()

		if self._list_names:
			print("")
			print("names.")
			for name in self._names:
				print(name)

		if self._list_errors:
			print("")
			print("errors.")
			self.print_errors()


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--list-names',
	is_flag=True,
	default=False,
	help="List found page names.")
@click.option(
	'--list-errors',
	is_flag=True,
	default=False)
@Processor.options
def stats(data_path, **kwargs):
	""" List stats of pages in DATA_PATH. """
	processor = StatsProcessor(kwargs)
	processor.traverse(data_path)
	processor.print()


if __name__ == "__main__":
	stats()
