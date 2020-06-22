import zipfile
import shapely
import json

from functools import lru_cache

from origami.core.page import Page
from origami.core.block import Block, Line


class LineLoader:
	def __init__(self, dewarped=False):
		self._dewarped = dewarped

	@lru_cache(maxsize=10)
	def _load_page(self, full_page_path):
		return Page(full_page_path, dewarp=self._dewarped)

	@lru_cache(maxsize=10)
	def _load_block(self, full_page_path, block_path):
		with zipfile.ZipFile(full_page_path.with_suffix(".contours.zip"), "r") as zf:
			name = block_path + ".wkt"
			polygon = shapely.wkt.loads(zf.read(name).decode("utf8"))
			return Block(self._load_page(full_page_path), polygon, dewarped=self._dewarped)

	def load_line(self, page_path, line_path):
		block = self._load_block(
			page_path, "/".join(line_path.split("/")[:3]))

		with zipfile.ZipFile(page_path.with_suffix(".lines.zip"), "r") as zf:
			line_info = json.loads(zf.read(line_path + ".json"))

		return Line(block, **line_info)

	def load_line_image(self, page_path, line_path, **kwargs):
		return self.load_line(page_path, line_path).image(dewarped=self._dewarped, **kwargs)
