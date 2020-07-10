#!/usr/bin/env python3

import sys
import time
import threading


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


# this nice Spinner class is taken from:
# https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor
class Spinner:
	busy = False
	delay = 0.1

	@staticmethod
	def spinning_cursor():
		while 1:
			for cursor in '|/-\\': yield cursor

	def __init__(self, delay=None):
		self.spinner_generator = self.spinning_cursor()
		if delay and float(delay): self.delay = delay

	def spinner_task(self):
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
		time.sleep(self.delay)
		if exception is not None:
			return False