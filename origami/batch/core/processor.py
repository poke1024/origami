#!/usr/bin/env python3

import os
import re
import click
import json
import logging
import zipfile
import portalocker
import contextlib
import traceback
import imghdr
import multiprocessing
import functools
import threading
import time
import sys

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


def is_image(path):
	# imghdr might be the perfect tool for this, but
	# it fails to detect some valid images. so we go
	# with extenstions for the most part.
	# see https://stackoverflow.com/questions/36870661/
	# imghdr-python-cant-detec-type-of-some-images-image-extension

	if path.suffix.lower() in (".jpg", ".png", ".tif", ".tiff"):
		return True

	return imghdr.what(path) is not None


class WatchdogState(enum.Enum):
	RUNNING = 0
	DONE = 1
	CANCEL = 2


class Watchdog(threading.Thread):
	def __init__(self, pool, timeout=1000):
		threading.Thread.__init__(self)
		self._pool = pool
		self._timeout = timeout
		self._last_ping = time.time()
		self._state = WatchdogState.RUNNING

	def _cancel(self, dt):
		if self._state != WatchdogState.CANCEL:
			logging.error("no new results after %d s. stopping." % dt)
			self._state = WatchdogState.CANCEL
			self._pool.terminate()
			t = threading.Thread(target=lambda: self._pool.join(), args=())
			t.start()
			self._last_ping = time.time()
		elif self._state == WatchdogState.CANCEL:
			logging.error("stopping failed. killing process.")
			os._exit(1)

	def run(self):
		while True:
			time.sleep(1)
			if self._state == WatchdogState.DONE:
				break
			dt = time.time() - self._last_ping
			if dt > self._timeout:
				self._cancel(dt)

	def ping(self):
		self._last_ping = time.time()

	def set_is_done(self):
		if self._state == WatchdogState.RUNNING:
			self._state = WatchdogState.DONE

	def is_cancelled(self):
		return self._state == WatchdogState.CANCEL


class Processor:
	def __init__(self, options, needs_qt=False):
		self._overwrite = options.get("overwrite", False)
		self._processes = options.get("processes", 1)
		self._timeout = options.get("alive", 300)
		self._name = options.get("name", "")
		self._verbose = False

		self._lock_strategy = options.get("lock_strategy", "DB")
		self._lock_level = options.get("lock_level", "PAGE")
		self._lock_timeout = options.get("lock_timeout", "60")
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

		self._print_paths = options.get("print_paths")
		self._plain = options.get("plain")
		if self._plain:
			self._print_paths = True

	@staticmethod
	def options(f):
		options = [
			click.option(
				'--processes',
				type=int,
				default=1,
				help="Number of parallel processes to employ."),
			click.option(
				'--alive',
				type=int,
				default=300,
				help="Seconds to wait after inactive process is killed."),
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
				'--lock-timeout',
				type=int,
				default=60,
				required=False,
				help="Timeout for locking. NFS volumes might need high values."),
			click.option(
				'--overwrite',
				is_flag=True,
				default=False,
				help="Recompute and overwrite existing result files."),
			click.option(
				'--profile',
				is_flag=True,
				default=False,
				help="Enable profiling and show results."),
			click.option(
				'--print-paths',
				is_flag=True,
				default=False,
				help="Print which files are processed."),
			click.option(
				'--plain',
				is_flag=True,
				default=False,
				help="Print plain output that is friendly to piping.")
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
		return item

	def _process_queue(self, queued):
		with self._profiler or nullcontext():
			squeue = sorted(queued)
			n = len(squeue)
			nd = len(str(n))

			if self._processes > 1:
				with multiprocessing.Pool(self._processes) as pool:
					watchdog = Watchdog(pool=pool, timeout=self._timeout)
					watchdog.start()

					with tqdm(total=len(squeue), disable=self._print_paths) as progress:
						for i, (p, kwargs) in enumerate(pool.imap_unordered(
							self._trigger_process_star, squeue)):

							if self._print_paths:
								print(f"[{str(i + 1).rjust(nd)}/{n}] {p}", flush=True)
							else:
								progress.update(1)

							watchdog.ping()

				if watchdog.is_cancelled():
					watchdog.kill()
					sys.exit(1)
				else:
					watchdog.set_is_done()
			elif self._print_paths:
				for i, (p, kwargs) in enumerate(squeue):
					self._trigger_process(p, kwargs)
					print(f"[{str(i + 1).rjust(nd)}/{n}] {p}", flush=True)
			else:
				for p, kwargs in tqdm(squeue):
					self._trigger_process(p, kwargs)

	def _build_queue(self, path):
		path = Path(path)
		if not path.exists():
			raise FileNotFoundError("%s does not exist." % path)

		queued = []
		counts = dict(images=0)

		def add_path(p):
			if not p.exists():
				print("skipping %s: path does not exist." % p)
				return

			if self._name and not re.search(self._name, str(p)):
				return

			if not is_image(p):
				if self._verbose:
					print("skipping %s: not an image." % p)
				return

			counts['images'] += 1

			if not self.should_process(p):
				if self._verbose:
					print("skipping %s: should_process is False" % p)
				return

			kwargs = self.prepare_process(p)
			if kwargs is not False:
				queued.append((p, kwargs))

		if not path.is_dir():
			if path.suffix == ".txt":
				with open(path, "r") as f:
					for line in f:
						line = line.strip()
						if line:
							add_path(Path(line))
			else:
				raise FileNotFoundError(
					"%s is not a valid path or text file of paths." % path)
		else:
			print(f"scanning {path}... ", flush=True, end="")

			with Spinner(disable=self._plain):
				for folder, _, filenames in os.walk(path):
					folder = Path(folder)
					if folder.name.endswith(".out"):
						continue

					for filename in filenames:
						add_path(folder / filename)

			print("done.", flush=True)
			print(f"{counts['images']} documents found, {len(queued)} ready to process.")

		return queued

	def traverse(self, path: Path):
		print(f"running {self.processor_name}.", flush=True)

		queued = self._build_queue(path)

		if self._lock_strategy == "DB":
			if self._lock_database:
				db_path = Path(self._lock_database)
			elif Path(path).is_dir():
				db_path = Path(path) / "origami.lock.db"
			else:
				db_path = Path(path).parent / "origami.lock.db"

			self._mutex = DatabaseMutex(
				db_path, timeout=self._lock_timeout)
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

			new_json_path = json_path.parent / (
				json_path.stem + ".updated" + json_path.suffix)
			if new_json_path.exists():
				os.remove(new_json_path)

			if json_path.exists():
				with open(json_path, "r") as f:
					file_data = f.read()
					data = json.loads(file_data)
			else:
				data = dict()

			for k, v in updates.items():
				data[k] = v

			with open(new_json_path, "w") as f:
				json.dump(data, f)

			if json_path.exists():
				os.remove(json_path)
			os.rename(new_json_path, json_path)

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
