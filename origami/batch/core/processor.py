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
import ctypes
import psutil

from pathlib import Path
from functools import partial
from tqdm import tqdm
from atomicwrites import atomic_write
from contextlib import contextmanager, nullcontext
from functools import partial

from origami.core.time import elapsed_timer
from origami.batch.core.io import *
from origami.batch.core.utils import Spinner
from origami.batch.core.mutex import DatabaseMutex, FileMutex, DummyMutex


def qt_app():
	try:
		from PySide2 import QtGui
	except ImportError:
		from PySide6 import QtGui

	os.environ["QT_QPA_PLATFORM"] = "offscreen"
	return QtGui.QGuiApplication()


class WatchdogState(enum.Enum):
	RUNNING = 0
	DONE = 1
	CANCEL = 2


class StopWatch:
	def __init__(self):
		self._last_reset = time.time()

	def reset(self):
		self._last_reset = time.time()

	@property
	def age(self):
		return time.time() - self._last_reset


class SharedMemoryStopWatch:
	def __init__(self):
		self._shared = multiprocessing.Value('L', int(time.time()))

	def reset(self):
		with self._shared.get_lock():
			self._shared.value = int(time.time())

	@property
	def age(self):
		with self._shared.get_lock():
			return time.time() - self._shared.value


WorkSetEntry = collections.namedtuple(
	'WorkSetEntry', ['path', 'pid', 'age'])


class SharedMemoryWorkSet:
	def __init__(self, access, n):
		assert n >= 1

		self._array = multiprocessing.Array(
			ctypes.c_int64, n * 4)

		# each slot has 4 integer entries:
		# 0: value
		# 1: pid
		# 2: timestamp
		# 3: not used

		self._n = n
		for i in range(self._n * 4):
			self._array[i] = -1

		self._access = access

	def _cleanup(self):
		with self._array.get_lock():
			for i in range(self._n):
				pid = self._array[4 * i + 1]
				if pid >= 0 and not psutil.pid_exists(pid):
					logging.warning(f"removing killed pid {pid} from work set.")
					self._array[4 * i] = -1
					self._array[4 * i + 1] = -1
					self._array[4 * i + 2] = -1

	def add(self, value):
		assert value >= 0
		with self._array.get_lock():
			self._cleanup()

			free = None
			for i in range(self._n):
				if self._array[4 * i] == value:
					return
				elif free is None and self._array[4 * i] < 0:
					free = i

			if free is None:
				raise RuntimeError(
					f"no free slots for adding {value}, pid {os.getpid()}: {self.active}")

			self._array[4 * free] = value
			self._array[4 * free + 1] = int(os.getpid())
			self._array[4 * free + 2] = int(time.time())

	def remove(self, value):
		assert value >= 0
		with self._array.get_lock():
			found = None
			for i in range(self._n):
				if self._array[4 * i] == value:
					found = i
					break
			assert found is not None
			self._array[4 * found] = -1
			self._array[4 * found + 1] = -1
			self._array[4 * found + 2] = -1

	@property
	def active(self):
		result = []
		with self._array.get_lock():
			self._cleanup()

			for i in range(self._n):
				if self._array[4 * i] >= 0:
					result.append(WorkSetEntry(
						path=self._access(self._array[4 * i]),
						pid=self._array[4 * i + 1],
						age=int(time.time() - self._array[4 * i + 2])))
		return result

	def print(self):
		active = self.active
		if active:
			logging.error(f"{len(active)} entries in work set:")
			for i, entry in enumerate(active):
				logging.error(f"  ({i + 1}) {entry}]")
		else:
			logging.error("no entries in work set.")


global global_stop_watch
global_stop_watch = SharedMemoryStopWatch()

# global_stop_watch needs to be global indeed, as pickling
# over the fork in imap_unordered will not work otherwise.

global global_work_set


class Watchdog(threading.Thread):
	def __init__(self, pool, stop_watch, work_set, timeout):
		threading.Thread.__init__(self)
		self._pool = pool
		self._timeout = timeout
		self._stop_watch = stop_watch
		self._work_set = work_set
		self._state = WatchdogState.RUNNING
		self._cond = threading.Condition()
		stop_watch.reset()

	def _print_work_set(self):
		self._work_set.print()

	def _cancel(self):
		if self._state != WatchdogState.CANCEL:
			logging.error("no new results after %d s. stopping." % self._stop_watch.age)
			self._print_work_set()
			self._state = WatchdogState.CANCEL
			self._pool.terminate()
			t = threading.Thread(target=lambda: self._pool.join(), args=())
			t.start()
			self._stop_watch.reset()
		elif self._state == WatchdogState.CANCEL:
			logging.error("stopping failed. killing process.")
			self._print_work_set()
			os._exit(1)

	def run(self):
		with self._cond:
			while True:
				self._cond.wait(
					max(0, self._timeout - self._stop_watch.age))

				if self._state == WatchdogState.DONE:
					break

				if self._stop_watch.age > self._timeout:
					self._cancel()

	def set_is_done(self):
		with self._cond:
			if self._state == WatchdogState.RUNNING:
				self._state = WatchdogState.DONE
				self._cond.notify()

	def is_cancelled(self):
		return self._state == WatchdogState.CANCEL


def chunks(items, n):
	for i in range(0, len(items), n):
		yield items[i:i + n]


class Processor:
	def __init__(self, options, needs_qt=False):
		self._overwrite = options.get("overwrite", False)
		self._processes = options.get("processes", 1)
		self._timeout = options.get("alive", 600)
		self._name = options.get("name", "")
		self._verbose = False

		self._lock_strategy = options.get("lock_strategy", "DB")
		self._lock_level = options.get("lock_level", "PAGE")
		self._lock_timeout = options.get("lock_timeout", "60")
		self._max_lock_age = options.get("max_lock_age")
		self._lock_chunk_size = 25
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

		self._print_paths = False
		self._plain = options.get("plain")
		if self._plain:
			self._print_paths = True

		self._debug_write = options.get("debug_write", False)
		self._track_changes = options.get("track_changes", False)

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
				default=600,
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
				help="Seconds to wait to acquire locking. NFS volumes might need high values."),
			click.option(
				'--max-lock-age',
				type=int,
				default=600,
				required=False,
				help="Maximum age of a lock in seconds until it is considered invalid."),
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
				'--plain',
				is_flag=True,
				default=False,
				help="Print plain output that is friendly to piping."),
			click.option(
				'--debug-write',
				is_flag=True,
				default=False,
				help="Debug which files are written."),
			click.option(
				'--track-changes',
				type=str,
				default="",
				help="Recompute files and track changes with given tag.")
		]
		return functools.reduce(lambda x, opt: opt(x), options, f)

	@property
	def processor_name(self):
		return self.__class__.__name__

	def is_image(self, path):
		# imghdr might be the perfect tool for this, but
		# it fails to detect some valid images. so we go
		# with extenstions for the most part.
		# see https://stackoverflow.com/questions/36870661/
		# imghdr-python-cant-detec-type-of-some-images-image-extension

		if path.suffix.lower() in (".jpg", ".png", ".tif", ".tiff"):
			return True

		return imghdr.what(path) is not None

	def should_process(self, page_path):
		return True

	def prepare_process(self, page_path):
		artifacts = self.artifacts()

		if self._track_changes:
			file_writer = TrackChangeWriter(self._track_changes)
		else:
			file_writer = AtomicFileWriter(overwrite=self._overwrite)
			if self._debug_write:
				file_writer = DebuggingFileWriter(file_writer)

		kwargs = dict()
		for arg, spec in artifacts:
			f = spec.instantiate(
				page_path=page_path,
				processor=self,
				file_writer=file_writer)

			f.fix_inconsistent()

			if not f.is_ready():
				if self._verbose:
					print("skipping %s: missing %s" % (page_path, f.missing))
				return False

			kwargs[arg] = f

		return kwargs

	def _trigger_process1(self, p, kwargs, locked):
		work = locked

		if not locked:
			logging.warning(f"failed to obtain lock for {p}. ignoring.")

		try:
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

	def _trigger_process(self, chunk):
		if self._lock_level == "PAGE":
			lock_actor_name = "page"
		elif self._lock_level == "TASK":
			lock_actor_name = self.processor_name
		else:
			raise ValueError(self._lock_level)

		with self._mutex.lock(
			lock_actor_name,
			[str(p) for _, p, _ in chunk]) as locked:

			for i, p, kwargs in chunk:
				global_work_set.add(i)
				try:
					self._trigger_process1(p, kwargs, locked)
				finally:
					global_work_set.remove(i)
				yield i, p

	def _trigger_process_async(self, chunk):
		results = []
		for i, p in self._trigger_process(chunk):
			results.append((i, p))
			global_stop_watch.reset()
		return results

	def _process_queue(self, queued):
		global global_work_set
		global_work_set = SharedMemoryWorkSet(
			lambda i: queued[i][1], max(1, self._processes))

		with self._profiler or nullcontext():
			chunked_queue_gen = chunks(queued, self._lock_chunk_size)

			def iprogress(i):
				nd = len(str(len(queued)))
				return f"[{str(i + 1).rjust(nd)} / {len(queued)}]"

			if self._processes > 1:
				with multiprocessing.Pool(self._processes, maxtasksperchild=4) as pool:
					watchdog = Watchdog(
						pool=pool,
						stop_watch=global_stop_watch,
						work_set=global_work_set,
						timeout=self._timeout)
					watchdog.start()

					with tqdm(total=len(queued), disable=self._print_paths) as progress:
						for chunk in pool.imap_unordered(
							self._trigger_process_async, chunked_queue_gen):

							if self._print_paths:
								for i, p in chunk:
									print(f"{iprogress(i)} {p}", flush=True)
							else:
								progress.update(len(chunk))

							global_stop_watch.reset()

				if watchdog.is_cancelled():
					watchdog.kill()
					sys.exit(1)
				else:
					watchdog.set_is_done()
			else:
				with tqdm(total=len(queued), disable=self._print_paths) as progress:
					for chunk in chunked_queue_gen:
						for i, p in self._trigger_process(chunk):
							if self._print_paths:
								print(f"{iprogress(i)} {p}", flush=True)
							else:
								progress.update(1)

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

			if not self.is_image(p):
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
				queued.append((len(queued), p, kwargs))

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
				for folder, dirs, filenames in os.walk(path):
					folder = Path(folder)
					if folder.name.endswith(".out"):
						dirs.clear()
						continue
					else:
						dirs.sort()

						for filename in sorted(filenames):
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

			self._mutex.clear_locks(self._max_lock_age)

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
				if v is None:
					del data[k]
				else:
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
