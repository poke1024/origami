import os
import re
import logging
import zipfile
import portalocker

from pathlib import Path
from tqdm import tqdm


class Processor:
	def traverse(self, path: Path):
		if not Path(path).is_dir():
			raise FileNotFoundError("%s is not a valid path." % path)

		queued = []

		for folder, _, filenames in os.walk(path):
			folder = Path(folder)
			for filename in filenames:
				p = folder / filename

				if p.name.endswith(".binarized.png"):
					continue
				if re.match(r".*\.debug\..*\.(png|jpg)$", p.name):
					continue

				if self.should_process(p):
					queued.append(p)

		for p in tqdm(queued):
			try:
				with self.page_lock(p) as _:
					self.process(p)
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
		return portalocker.Lock(path, "r", flags=portalocker.LOCK_EX, timeout=1)

	def lock(self, path, mode):
		return portalocker.Lock(path, mode, flags=portalocker.LOCK_EX, timeout=1)

	@property
	def compression(self):
		return zipfile.ZIP_DEFLATED  # zipfile.ZIP_LZMA
