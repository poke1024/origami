#!/usr/bin/env python3

import sys
import time
import threading
import collections
import shapely
from itertools import chain


class RegionsFilter:
	def __init__(self, spec):
		self._paths = set()
		for s in spec.split(","):
			self._paths.add(
				tuple(s.strip().split("/")))

	def __call__(self, path):
		return tuple(path[:2]) in self._paths

	@property
	def paths(self):
		return list(self._paths)


class TableRegionCombinator:
	""" reverses the splitting done in subdivide_table_blocks """

	def __init__(self, paths):
		mapping = collections.defaultdict(list)
		for k in paths:
			parts = k[-1].split(".")
			if len(parts) > 1:
				mapping[k[:-1] + (parts[0], )].append(k)
			else:
				mapping[k].append(k)
		self._mapping = mapping

	@property
	def mapping(self):
		return self._mapping

	def contours_from_blocks(self, blocks):
		contours = dict([
			(k, b.image_space_polygon)
			for k, b in blocks.items()])
		return self.contours(contours)

	def contours(self, contours):
		combined = dict()
		for k, v in self._mapping.items():
			if len(v) == 1:
				combined[k] = contours[v[0]]
			else:
				geom = shapely.ops.cascaded_union([
					contours[x] for x in v])
				if geom.geom_type != "Polygon":
					geom = geom.convex_hull
				combined[k] = geom
		return combined

	def lines(self, lines):
		lines_by_block = collections.defaultdict(list)
		for k, line in lines.items():
			lines_by_block[k[:3]].append(line)

		combined = dict()
		for k, v in self._mapping.items():
			combined[k] = list(chain(
				*[lines_by_block[x] for x in v]))

		new_lines = dict()
		for k, v in combined.items():
			for i, line in enumerate(v):
				new_lines[k + (1 + i,)] = line

		return new_lines


# this nice Spinner class is taken from:
# https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor
class Spinner:
	@staticmethod
	def spinning_cursor():
		while True:
			for cursor in '|/-\\':
				yield cursor

	def __init__(self, delay=0.1, disable=False):
		if not disable:
			self.spinner_generator = self.spinning_cursor()
		else:
			self.spinner_generator = None

		self.delay = delay
		self.busy = True

	def spinner_task(self):
		if self.spinner_generator:
			while self.busy:
				sys.stdout.write(next(self.spinner_generator))
				sys.stdout.flush()
				time.sleep(self.delay)
				sys.stdout.write('\b')
				sys.stdout.flush()

	def __enter__(self):
		self.busy = True
		threading.Thread(target=self.spinner_task).start()

	def __exit__(self, exception, value, tb):
		self.busy = False
		if self.spinner_generator:
			time.sleep(self.delay)
		if exception is not None:
			return False
