import zipfile

from origami.batch.core.processor import Processor
from origami.batch.core.utils import *
from origami.core.segment import Segmentation


class BlockProcessor(Processor):
	def read_contours(self, page_path: Path, pred_type, stage=Stage.WARPED):
		return read_contours(page_path, pred_type, stage=stage, open=self.lock)

	def read_blocks(self, page_path: Path, stage=Stage.WARPED):
		return read_blocks(page_path, stage=stage, open=self.lock)

	def read_dewarped_blocks(self, page_path):
		return self.read_blocks(page_path, stage=Stage.DEWARPED)

	def read_aggregate_blocks(self, page_path):
		return self.read_blocks(page_path, stage=Stage.AGGREGATE)

	def read_separators(self, page_path: Path):
		return read_separators(page_path, stage=Stage.WARPED, open=self.lock)

	def read_dewarped_separators(self, page_path: Path):
		return read_separators(page_path, stage=Stage.DEWARPED, open=self.lock)

	def read_lines(self, page_path: Path, blocks, stage=Stage.WARPED):
		return read_lines(page_path, blocks, stage=stage, open=self.lock)

	def read_aggregate_lines(self, page_path: Path, blocks):
		return self.read_lines(page_path, blocks, stage=Stage.AGGREGATE)

	def read_predictors(self, page_path: Path):
		return Segmentation.read_predictors(page_path.with_suffix(".segment.zip"))

	def read_reliable_contours(self, page_path: Path):
		return read_reliable_contours(page_path)

