#!/usr/bin/env python3

import os
import re
import click
import json
import logging
import zipfile
import portalocker
import contextlib
import contextlib
import traceback
import imghdr
import multiprocessing
import functools

from pathlib import Path
from functools import partial
from tqdm import tqdm
from atomicwrites import atomic_write
from contextlib import contextmanager, nullcontext

from origami.core.time import elapsed_timer
from origami.batch.core.io import *
from origami.batch.core.utils import Spinner
from origami.batch.core.mutex import DatabaseMutex, FileMutex, DummyMutex


def qt_app():
	from PySide2 import QtGui
	os.environ["QT_QPA_PLATFORM"] = "offscreen"
	return QtGui.QGuiApplication()


class Processor:
	def __init__(self, options, needs_qt=False):
		self._overwrite = options.get("overwrite", False)
		self._processes = options.get("processes", 1)
		self._name = options.get("name", "")
		self._verbose = False

		self._lock_strategy = options.get("lock_strategy", "DB")
		self._lock_level = options.get("lock_level", "PAGE")
		self._mutex = None

		if self._lock_strategy == "DB":
			self._lock_database = options.get("lock_database")
		elif self._lock_strategy in ("FILE", "NONE"):
			pass
		else:
			raise ValueError(self._lock_strategy)

		if needs_qt:
			self._qt_app = qt_app()
			if self._processes > 1:
				logging.warning(
					"this batch does not support multiple processes.")
				self._processes = 1  # cannot safely fork here.
		else:
			self._qt_app = None

		if options.get("profile"):
			from profiling.sampling import SamplingProfiler
			self._profiler = SamplingProfiler()
			self._overwrite = True  # profile implies overwrite
		else:
			self._profiler = None

	@staticmethod
	def options(f):
		options = [
			click.option(
				'--processes',
				type=int,
				default=1,
				help="Number of parallel processes to employ."),
			click.option(
				'--name',
				type=str,
				default="",
				help="Only process paths that conform to the given pattern."),
			click.option(
				'--lock-strategy',
				type=click.Choice(['FILE', 'DB', 'NONE'], case_sensitive=False),
				default="DB",
				help="How to implement locking for concurrency."),
			click.option(
				'--lock-level',
				type=click.Choice(['PAGE', 'TASK'], case_sensitive=False),
				default="PAGE",
				help="Lock granularity."),
			click.option(
				'--lock-database',
				type=click.Path(),
				required=False,
				help="Mutex database path used for concurrent processing"),
			click.option(
				'--overwrite',
				is_flag=True,
				default=False,
				help="Recompute and overwrite existing result files."),
			click.option(
				'--profile',
				is_flag=True,
				default=False,
				help="Enable profiling and show results.")
		]
		return functools.reduce(lambda x, opt: opt(x), options, f)

	@property
	def processor_name(self):
		return self.__class__.__name__

	def should_process(self, page_path):
		return True

	def prepare_process(self, page_path):
		artifacts = self.artifacts()

		kwargs = dict()
		for arg, spec in artifacts:
			f = spec.instantiate(
				page_path=page_path,
				processor=self,
				overwrite=self._overwrite)

			if not f.is_ready():
				if self._verbose:
					print("skipping %s: missing " % (page_path, f.missing))
				return False

			kwargs[arg] = f

		return kwargs

	def _trigger_process(self, p, kwargs):
		try:
			if self._lock_level == "PAGE":
				lock_actor_name = "page"
			elif self._lock_level == "TASK":
				lock_actor_name = self.processor_name
			else:
				raise ValueError(self._lock_level)

			with self._mutex.lock(lock_actor_name, str(p)) as locked:
				work = locked

				if work:
					# a concurrent worker might already have done this.
					for f in kwargs.values():
						if not f.is_ready():
							work = False
							break

				if work:
					with elapsed_timer() as elapsed:
						data_path = find_data_path(p)
						data_path.mkdir(exist_ok=True)

						runtime_info = self.process(p, **kwargs)

					if runtime_info is None:
						runtime_info = dict()
					runtime_info["status"] = "COMPLETED"
					runtime_info["elapsed"] = round(elapsed(), 2)

					self._update_runtime_info(
						p, {self.processor_name: runtime_info})

		except KeyboardInterrupt:
			logging.exception("Interrupted at %s." % p)
			raise
		except:
			logging.exception("Failed to process %s." % p)
			runtime_info = dict(
				status="FAILED",
				traceback=traceback.format_exc())
			self._update_runtime_info(p, {
				self.processor_name: runtime_info})
		finally:
			# free memory allocated in cached io.Reader
			# attributes. this can get substantial for
			# long runs.
			kwargs.clear()

	def _trigger_process_star(self, item):
		self._trigger_process(*item)

	def _process_queue(self, queued):
		with self._profiler or nullcontext():
			squeue = sorted(queued)

			if self._processes > 1:
				with multiprocessing.Pool(self._processes) as pool:
					with tqdm(total=len(squeue)) as progress:
						for _ in pool.imap_unordered(
							self._trigger_process_star, squeue):
							progress.update(1)
			else:
				for p, kwargs in tqdm(squeue):
					self._trigger_process(p, kwargs)

	def _build_queue(self, path):
		queued = []

		if not Path(path).is_dir():
			raise FileNotFoundError("%s is not a valid path." % path)

		print("scanning data path... ", flush=True, end="")

		with Spinner():
			for folder, _, filenames in os.walk(path):
				folder = Path(folder)
				if folder.name.endswith(".out"):
					continue

				for filename in filenames:
					p = folder / filename

					if self._name and not re.search(self._name, str(p)):
						continue
					if imghdr.what(p) is None:
						if self._verbose:
							print("skipping %s: not an image." % p)
						continue

					if not self.should_process(p):
						if self._verbose:
							print("skipping %s: should_process is False" % p)
						continue

					kwargs = self.prepare_process(p)
					if kwargs is not False:
						queued.append((p, kwargs))

		print("done.", flush=True)

		return queued

	def traverse(self, path: Path):
		queued = self._build_queue(path)

		if self._lock_strategy == "DB":
			if self._lock_database:
				db_path = Path(self._lock_database)
			else:
				db_path = Path(path) / "origami.lock.db"

			self._mutex = DatabaseMutex(db_path)
		elif self._lock_strategy == "FILE":
			self._mutex = FileMutex()
		elif self._lock_strategy == "NONE":
			self._mutex = DummyMutex()
		else:
			raise ValueError(self._lock_strategy)

		try:
			self._process_queue(queued)
		finally:
			self._mutex = None

		if self._profiler:
			self._profiler.run_viewer()

	def process(self, p: Path):
		pass

	def lock_or_open(self, path, mode):
		if self._lock_strategy == "FILE":
			return portalocker.Lock(
				path,
				mode,
				flags=portalocker.LOCK_EX,
				timeout=1,
				fail_when_locked=True)
		else:
			return open(path, mode)

	def _update_json(self, page_path, artifact, updates):
		try:
			data_path = find_data_path(page_path)
			json_path = data_path / artifact.filename()

			if not json_path.exists():
				mode = "w+"
			else:
				mode = "r+"

			with open(json_path, mode) as f:
				f.seek(0)
				file_data = f.read()
				if file_data:
					data = json.loads(file_data)
				else:
					data = dict()
				for k, v in updates.items():
					data[k] = v

				f.seek(0)
				json.dump(data, f)
				f.truncate()
		except:
			logging.error(traceback.format_exc())

	def _update_runtime_info(self, page_path, updates):
		self._update_json(page_path, Artifact.RUNTIME, updates)

	@property
	def compression(self):
		return zipfile.ZIP_DEFLATED  # zipfile.ZIP_LZMA

	@contextmanager
	def write_zip_file(self, path, overwrite=False):
		with atomic_write(path, mode="wb", overwrite=overwrite) as f:
			with zipfile.ZipFile(f, "w", self.compression) as zf:
				yield zf
