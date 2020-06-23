from origami.batch.core.processor import Processor
from origami.batch.core.utils import *
from origami.api import Segmentation


class BlockProcessor(Processor):
	def read_contours(self, page_path: Path, pred_type):
		return read_contours(page_path, pred_type, open=self.lock)

	def read_blocks(self, page_path: Path, dewarped=False):
		return read_blocks(page_path, dewarped=dewarped, open=self.lock)

	def read_dewarped_blocks(self, page_path):
		return self.read_blocks(page_path, dewarped=True)

	def read_separators(self, page_path: Path):
		return read_separators(page_path, open=self.lock)

	def read_lines(self, page_path: Path, blocks, dewarped=False):
		return read_lines(page_path, blocks, dewarped=dewarped, open=self.lock)

	def read_dewarped_lines(self, page_path: Path, blocks):
		return self.read_lines(page_path, blocks, dewarped=True)

	def read_predictors(self, page_path: Path):
		return Segmentation.read_predictors(page_path.with_suffix(".segment.zip"))
