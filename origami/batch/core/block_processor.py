from origami.batch.core.processor import Processor
from origami.batch.core.utils import *


class BlockProcessor(Processor):
	def read_contours(self, page_path: Path, pred_type):
		return read_contours(page_path, pred_type, open=self.lock)

	def read_blocks(self, page_path: Path):
		return read_blocks(page_path, open=self.lock)

	def read_separators(self, page_path: Path):
		return read_separators(page_path, open=self.lock)

	def read_lines(self, page_path: Path, blocks):
		return read_lines(page_path, blocks, open=self.lock)
