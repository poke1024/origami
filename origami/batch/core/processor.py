#!/usr/bin/env python3

import os
import re
import json
import logging
import zipfile
import portalocker
import contextlib
import traceback
import imghdr

from pathlib import Path
from functools import partial
from tqdm import tqdm
from atomicwrites import atomic_write
from contextlib import contextmanager, nullcontext

from origami.core.time import elapsed_timer
from origami.batch.core.io import *


class Processor:
	def __init__(self, options):
		self._lock_files = not options.get("nolock", True)
		self._overwrite = options.get("overwrite", False)
		self._name = options.get("name", "")

		if options.get("profile"):
			from profiling.sampling import SamplingProfiler
			self._profiler = SamplingProfiler()
			self._overwrite = True  # profile implies overwrite
		else:
			self._profiler = None

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
				return False

			kwargs[arg] = f

		return kwargs

	def traverse(self, path: Path):
		if not Path(path).is_dir():
			raise FileNotFoundError("%s is not a valid path." % path)

		queued = []

		for folder, _, filenames in os.walk(path):
			folder = Path(folder)
			if folder.name.endswith(".out"):
				continue

			for filename in filenames:
				p = folder / filename

				if self._name and not re.search(self._name, str(p)):
					continue
				if imghdr.what(p) is None:
					continue

				if not self.should_process(p):
					continue

				kwargs = self.prepare_process(p)
				if kwargs is not False:
					queued.append((p, kwargs))

		with self._profiler or nullcontext():
			for p, kwargs in tqdm(sorted(queued)):
				try:
					with self.page_lock(p) as _:
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
					break
				except:
					logging.exception("Failed to process %s." % p)
					runtime_info = dict(
						status="FAILED",
						traceback=traceback.format_exc())
					self._update_runtime_info(p, {
						self.processor_name: runtime_info})

		if self._profiler:
			self._profiler.run_viewer()

	def process(self, p: Path):
		pass

	def page_lock(self, path):
		if self._lock_files:
			return portalocker.Lock(path, "r", flags=portalocker.LOCK_EX, timeout=1)
		else:
			return contextlib.nullcontext()

	def lock(self, path, mode):
		if self._lock_files:
			return portalocker.Lock(path, mode, flags=portalocker.LOCK_EX, timeout=1)
		else:
			return contextlib.nullcontext()

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
