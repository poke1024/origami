#!/usr/bin/python

import os
import datetime
import logging

from lxml import etree
from pathlib import Path
from cached_property import cached_property


namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
nsmap = {None: namespace}


class Document:
	def __init__(self, filename, image_size):
		root = etree.Element(etree.QName(namespace, "PcGts"), nsmap=nsmap)

		metadata = etree.Element(etree.QName(namespace, "Metadata"), nsmap=nsmap)
		creator = etree.Element(etree.QName(namespace, "Creator"), nsmap=nsmap)
		creator.text = "Origami"
		metadata.append(creator)
		created = etree.Element(etree.QName(namespace, "Created"), nsmap=nsmap)
		created.text = datetime.datetime.now().isoformat()
		metadata.append(created)
		last_change = etree.Element(etree.QName(namespace, "LastChange"), nsmap=nsmap)
		last_change.text = datetime.datetime.now().isoformat()
		metadata.append(last_change)
		root.append(metadata)

		page = etree.Element(etree.QName(namespace, "Page"), nsmap=nsmap)
		page.set('imageFilename', filename)
		page.set('imageWidth', "%d" % image_size[0])
		page.set('imageHeight', "%d" % image_size[1])
		root.append(page)

		self._root = root
		self._page = page

	def append(self, element):
		self._page.append(element._node)

	def remove(self, element):
		self._page.remove(element._node)

	def append_region(self, class_, **kwargs):
		region = Region(class_=class_, **kwargs)
		self.append(region)
		return region

	def append_text_region(self, **kwargs):
		return self.append_region(class_="TextRegion", **kwargs)

	def append_reading_order(self):
		ro = ReadingOrder()
		self.append(ro)
		return ro

	def write(self, path, validate=True, overwrite=False):
		if not overwrite and Path(path).exists():
			raise ValueError("xml file at %s already exists" % path)
		if isinstance(path, Path):
			path = str(path)
		tree = etree.ElementTree(self._root)
		tree.write(
			path,
			encoding='utf-8',
			xml_declaration=True,
			pretty_print=True)
		if validate:
			self.validate(doc=tree)

	@cached_property
	def xml_schema(self):
		script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
		with open(script_dir / "pagecontent.xsd", "r") as f:
			xmlschema_doc = etree.parse(f)
		return etree.XMLSchema(xmlschema_doc)

	def validate(self, path=None, doc=None):
		if doc is None:
			with open(path, "r") as f:
				doc = etree.parse(f)
		try:
			self.xml_schema.assertValid(doc)
		except etree.DocumentInvalid as e:
			log = self.xml_schema.error_log
			logging.error(log.last_error)
			raise


def format_coord(p):
	return '%d,%d' % tuple(map(round, p))


def make_coords_node(coords):
	coords_str = ' '.join(format_coord(p) for p in coords)
	node = etree.Element(etree.QName(namespace, "Coords"), nsmap=nsmap)
	node.set("points", coords_str)
	return node


def make_text_node(text):
	unicode_node = etree.Element(
		etree.QName(namespace, "Unicode"), nsmap=nsmap)
	unicode_node.text = text

	text_equiv_node = etree.Element(
		etree.QName(namespace, "TextEquiv"), nsmap=nsmap)
	text_equiv_node.append(unicode_node)
	return text_equiv_node


class ReadingOrder:
	def __init__(self):
		self._node = etree.Element(
			etree.QName(namespace, "ReadingOrder"), nsmap=nsmap)

	def append_ordered_group(self, **kwargs):
		g = OrderedGroup(**kwargs)
		self._node.append(g._node)
		return g


class OrderedGroup:
	def __init__(self, id_, caption=""):
		self._node = etree.Element(
			etree.QName(namespace, "OrderedGroup"), nsmap=nsmap)
		self._node.set("id", id_)
		if caption:
			self._node.set("caption", caption)

	def append_region_ref_indexed(self, index, region_ref):
		node = etree.Element(
			etree.QName(namespace, "RegionRefIndexed"), nsmap=nsmap)
		node.set("index", str(index))
		node.set("regionRef", region_ref)
		self._node.append(node)


class Region:
	def __init__(self, id_, class_="TextRegion", type_=None):
		self._node = etree.Element(
			etree.QName(namespace, class_), nsmap=nsmap)
		self._node.set('id', id_)
		if type_ is not None:
			self._node.set('type', type_)

	def append_coords(self, coords):
		self._node.append(make_coords_node(coords))

	def prepend_coords(self, coords):
		self._node.insert(0, make_coords_node(coords))

	def append_text_equiv(self, text):
		self._node.append(make_text_node(text))

	def append(self, element):
		self._node.append(element._node)

	def remove(self, element):
		self._node.remove(element._node)

	def append_text_line(self, **kwargs):
		line = TextLine(**kwargs)
		self.append(line)
		return line

	def append_text_region(self, **kwargs):
		region = Region(class_="TextRegion", **kwargs)
		self.append(region)
		return region


class TextRegion(Region):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs, class_="TextRegion")


class TextLine:
	def __init__(self, id_):
		self._node = etree.Element(
			etree.QName(namespace, "TextLine"), nsmap=nsmap)
		self._node.set('id', id_)

	def append_coords(self, coords):
		self._node.append(make_coords_node(coords))

	def append_text_equiv(self, text):
		self._node.append(make_text_node(text))
