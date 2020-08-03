#!/usr/bin/env python3

import click
import sqlite3


@click.command()
@click.argument(
	'db-path',
	type=click.Path(exists=True))
@click.option(
	'--with-db',
	type=click.Path(exists=True),
	help="DB to merge with.",
	required=True)
def merge_db(db_path, with_db):
	""" Merge database with other database. """
	conn_src = sqlite3.connect(with_db)
	cursor = conn_src.cursor()
	cursor.execute("SELECT * FROM lines ORDER BY page_path, line_path")
	data = cursor.fetchall()
	cursor.close()

	'''
	ATTACH DATABASE 'other.db' AS other;

	INSERT INTO other.tbl
	SELECT * FROM main.tbl;
	'''

	print(data[0])

	dst_conn = sqlite3.connect(db_path)
	with dst_conn:
		pass


if __name__ == "__main__":
	merge_db()
