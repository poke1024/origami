#!/usr/bin/python

import os
import datetime

import xml.etree.ElementTree as ET

from lxml import etree
from pathlib import Path


class Document:
	def __init__(self, filename, image_size):
		root = ET.Element('PcGts')
		root.set('xmlns', 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15')
		root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
		root.set('xsi:schemaLocation', 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15')

		metadata = ET.Element("Metadata")
		creator = ET.Element("Creator")
		creator.text = "Origami"
		metadata.append(creator)
		created = ET.Element("Created")
		created.text = datetime.datetime.now().isoformat()
		metadata.append(created)
		last_change = ET.Element("LastChange")
		last_change.text = datetime.datetime.now().isoformat()
		metadata.append(last_change)
		root.append(metadata)

		page = ET.Element('Page')
		page.set('imageFilename', filename)
		page.set('imageWidth', "%d" % image_size[0])
		page.set('imageHeight', "%d" % image_size[1])
		root.append(page)

		self._root = root
		self._page = page

	def append(self, element):
		self._page.append(element._node)

	def write(self, path, validate=True, overwrite=False):
		if not overwrite and Path(path).exists():
			raise ValueError("xml file at %s already exists" % path)
		ET.ElementTree(self._root).write(
			path, encoding='utf-8', xml_declaration=True)
		if validate:
			self.validate(path)

	def validate(self, path):
		script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
		with open(script_dir / "pagecontent.xsd", "r") as f:
			xmlschema_doc = etree.parse(f)
		xmlschema = etree.XMLSchema(xmlschema_doc)

		with open(path, "r") as f:
			doc = etree.parse(f)
		xmlschema.assertValid(doc)


def format_coord(p):
	return '%d,%d' % tuple(map(round, p))


def make_coords_node(coords):
	coords_str = ' '.join(format_coord(p) for p in coords)
	node = ET.Element('Coords')
	node.set("points", coords_str)
	return node


def make_text_node(text):
	unicode_node = ET.Element('Unicode')
	unicode_node.text = text

	text_equiv_node = ET.Element('TextEquiv')
	text_equiv_node.append(unicode_node)
	return text_equiv_node


class TextRegion:
	def __init__(self, id_, type_=None, primary_language="German"):
		self._node = ET.Element('TextRegion')
		self._node.set('id', id_)
		if type_ is not None:
			self._node.set('type', type_)
		self._node.set('primaryLanguage', primary_language)

	def append_coords(self, coords):
		self._node.append(make_coords_node(coords))

	def append_text_equiv(self, text):
		self._node.append(make_text_node(text))

	def append(self, element):
		self._node.append(element._node)


class TextLine:
	def __init__(self, id_):
		self._node = ET.Element('TextLine')
		self._node.set('id', id_)

	def append_coords(self, coords):
		self._node.append(make_coords_node(coords))
