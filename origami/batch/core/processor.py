import os
import re
import json
import logging
import zipfile
import portalocker
import contextlib
import traceback

from pathlib import Path
from tqdm import tqdm

from origami.core.time import elapsed_timer


class Processor:
	def __init__(self, options):
		self._lock_files = not options["nolock"]
		self._name = options.get("name", "")

	@property
	def processor_name(self):
		return self.__class__.__name__

	def traverse(self, path: Path):
		if not Path(path).is_dir():
			raise FileNotFoundError("%s is not a valid path." % path)

		queued = []

		for folder, _, filenames in os.walk(path):
			folder = Path(folder)
			for filename in filenames:
				p = folder / filename

				if re.match(r".*\.annotate\..*\.(png|jpg)$", p.name):
					continue
				if self._name and not re.search(self._name, str(p)):
					continue

				if self.should_process(p):
					queued.append(p)

		for p in tqdm(sorted(queued)):
			try:
				with self.page_lock(p) as _:
					with elapsed_timer() as elapsed:
						runtime_info = self.process(p)

					if runtime_info is None:
						runtime_info = dict()
					runtime_info["total_time"] = round(elapsed(), 2)

					self._update_runtime_info(
						p, {self.processor_name: runtime_info})

			except KeyboardInterrupt:
				logging.exception("Interrupted at %s." % p)
				break
			except:
				logging.exception("Failed to process %s." % p)

	def should_process(self, p: Path) -> bool:
		return True

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

	def _update_runtime_info(self, path, updates):
		try:
			json_path = path.with_suffix(".runtime.json")
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

	@property
	def compression(self):
		return zipfile.ZIP_DEFLATED  # zipfile.ZIP_LZMA
