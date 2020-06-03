import click
import sqlite3


@click.command()
@click.argument(
	'db-path',
	type=click.Path(exists=True))
def migrate_db(db_path):
	""" Migrate specified database to new format. """
	conn = sqlite3.connect(db_path)
	with conn:
		conn.execute('ALTER TABLE lines ADD COLUMN training BOOLEAN DEFAULT TRUE')
		conn.execute('ALTER TABLE lines ADD COLUMN validation BOOLEAN DEFAULT TRUE')


if __name__ == "__main__":
	migrate_db()
