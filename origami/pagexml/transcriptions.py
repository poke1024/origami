import xml.etree.ElementTree as etree
import shapely.strtree
import logging


class TranscriptionReader:
	def __init__(self, path):
		self._path = path
		self._root = etree.parse(path).getroot()

		namespaces = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}

		polygons = []
		for text_line in self._root.findall(".//page:TextLine", namespaces):
			points = text_line.find(".//page:Coords", namespaces).get("points")
			text_unicode = text_line.find(".//page:TextEquiv//page:Unicode", namespaces)
			if text_unicode is not None and text_unicode.text is not None:
				text = text_unicode.text.strip()

				if text:
					polygon = shapely.geometry.Polygon(
						[tuple(map(int, pt.split(","))) for pt in points.split()])
					polygon = polygon.buffer(0)  # fix invalid polygons.

					# we use polygon name to attach transcription data, we later retrieve it
					# in TranscriptionReader.get_text().
					polygon.name = text

					polygons.append(polygon)

		self._tree = shapely.strtree.STRtree(polygons)
		self._polygons = polygons

		self._fetched = dict()
		self._notfound = []

	def _get_polygon(self, line):
		line_polygon = line.image_space_polygon
		line_area = line_polygon.area

		candidates = list(self._tree.query(line_polygon))

		best_area = 0
		best_candidate = None
		for candidate in candidates:
			area = line_polygon.intersection(candidate).area
			if area > best_area and (area / line_area > 0.6 or line_polygon.contains(candidate.buffer(-3))):
				best_candidate = candidate
				best_area = area

		return best_candidate

	def get_text(self, line):
		polygon = self._get_polygon(line)
		if not polygon:
			return None
		return polygon.name

	def fetch_texts(self, lines, threshold=0.6):
		dst_polygons = []
		for i, line in enumerate(lines):
			p = shapely.wkt.loads(line.image_space_polygon.wkt)
			p.name = str(i)
			dst_polygons.append(p)
		tree = shapely.strtree.STRtree(dst_polygons)

		src_polygons = self._polygons

		areas = []
		for i, polygon in enumerate(src_polygons):
			if polygon.area < 10:
				continue
			candidates = list(tree.query(polygon))
			for candidate in candidates:
				area = polygon.intersection(candidate).area / polygon.area
				if area > threshold:
					j = int(candidate.name)
					areas.append((i, j, area))

		areas = sorted(areas, key=lambda x: x[-1], reverse=True)
		matched = set()
		rmatch = dict()
		duplicates = []
		for i, j, _ in areas:
			if i not in matched:
				if j in rmatch:
					duplicates.append((i, rmatch[j], j))
				else:
					lines[j].set_text(src_polygons[i].name)
					rmatch[j] = i
				matched.add(i)

		if duplicates:
			logging.warning("found %d duplicate matchings in %s. details in conflict.txt." % (
				len(duplicates), self._path))
			with open(self._path.parent / (self._path.stem + ".conflict.txt"), "w") as f:
				for i1, i2, j in duplicates:
					f.write("%s\n" % src_polygons[i1].wkt)
					f.write("%s\n" % src_polygons[i2].wkt)
					f.write("%s\n" % dst_polygons[j].wkt)

		'''
			line_area = polygon.area
			polygon_body = polygon.buffer(-3)

			best_area = 0
			best_candidate = None
			for candidate in candidates:
				area = polygon.intersection(candidate).area
				if area > best_area and (area / line_area > 0.6 or candidate.contains(polygon_body)):
					best_candidate = candidate
					best_area = area

			if best_candidate:
				already_matched = match.get(int(best_candidate.name))

				if already_matched:
					with open(self._path.parent / (self._path.stem + ".conflict.txt"), "w") as f:
						f.write("%s\n" % best_candidate.wkt)
						f.write("%s\n" % src_polygons[already_matched].wkt)
						f.write("%s\n" % src_polygons[i].wkt)
					raise RuntimeError("line matched more than once. details written to conflict.txt.")

				match[int(best_candidate.name)] = i
			else:
				unfetched.append(polygon.wkt)

		with open(self._path.parent / (self._path.stem + ".unfetched.txt"), "w") as f:
			for wkt in unfetched:
				f.write("%s\n" % wkt)

		for j, i in match.items():
			lines[j].set_text(src_polygons[i].name)
		'''

	def log_unfetched(self):
		with open(self._path.parent / (self._path.stem + ".unfetched.txt"), "w") as f:
			for p in self._polygons:
				if id(p) not in self._fetched:
					f.write("%s\n" % p.wkt)

		with open(self._path.parent / (self._path.stem + ".notfound.txt"), "w") as f:
			for wkt in self._notfound:
				f.write("%s\n" % wkt)

		#print("!", len([p for p in self._polygons if id(p) not in self._fetched]))
