#!/usr/bin/env python3


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
