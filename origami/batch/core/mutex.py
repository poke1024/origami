import sqlalchemy
import datetime
import os
import portalocker
import logging

from pathlib import Path
from contextlib import contextmanager


class DatabaseMutex:
	def __init__(self, path, timeout=1):
		self._db_uri = 'sqlite:///%s' % str(Path(path))
		self._timeout = timeout

		self._engine = None
		self._metadata = None
		self._mutex_table = None
		self._connect()

		try:
			self._metadata.create_all()
		except sqlalchemy.exc.OperationalError as e:
			# ignore this, usually this is an error inside sqlalchemy, see
			# https://github.com/sqlalchemy/sqlalchemy/issues/4936
			logging.exception("Metadata creation failed.")

	def __getstate__(self):
		return dict(db_uri=self._db_uri, timeout=self._timeout)

	def __setstate__(self, newstate):
		self._db_uri = newstate["db_uri"]
		self._timeout = newstate["timeout"]
		self._engine = None
		self._metadata = None
		self._mutex_table = None
		self._connect()

	def _connect(self):
		engine = sqlalchemy.create_engine(
			self._db_uri,
			isolation_level="SERIALIZABLE",
			poolclass=sqlalchemy.pool.NullPool,
			connect_args={"timeout": self._timeout})

		# see https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#pysqlite-serializable
		@sqlalchemy.event.listens_for(engine, "connect")
		def do_connect(dbapi_connection, connection_record):
			# disable pysqlite's emitting of the BEGIN statement entirely.
			# also stops it from emitting COMMIT before any DDL.
			dbapi_connection.isolation_level = None

		@sqlalchemy.event.listens_for(engine, "begin")
		def do_begin(conn):
			conn.execute("BEGIN EXCLUSIVE")

		self._engine = engine

		self._metadata = sqlalchemy.MetaData(self._engine)

		self._mutex_table = sqlalchemy.Table(
			"mutex", self._metadata,
			sqlalchemy.Column('path', sqlalchemy.Text, nullable=False),
			sqlalchemy.Column('processor', sqlalchemy.Text, nullable=False),
			sqlalchemy.Column('pid', sqlalchemy.BigInteger, nullable=False),
			sqlalchemy.Column('time', sqlalchemy.DateTime, nullable=False),
			sqlalchemy.PrimaryKeyConstraint('path', 'processor', name='mutex_pk'))

	def try_lock(self, processor, path):
		conn = self._engine.connect()

		try:
			conn.execute(self._mutex_table.insert(), [
				dict(
					path=path,
					processor=processor,
					pid=os.getpid(),
					time=datetime.datetime.now()
				)
			])

			locked = True
		except sqlalchemy.exc.IntegrityError as e:
			locked = False
		finally:
			conn.close()

		return locked

	def unlock(self, processor, path):
		conn = self._engine.connect()

		try:
			table = self._mutex_table
			stmt = table.delete().where(sqlalchemy.and_(
				table.c.processor == processor,
				table.c.path == path,
				table.c.pid == os.getpid()))
			conn.execute(stmt)
		finally:
			conn.close()

	@contextmanager
	def lock(self, processor, path):
		success = self.try_lock(processor, path)
		try:
			yield success
		finally:
			if success:
				self.unlock(processor, path)


class FileMutex:
	@contextmanager
	def lock(self, processor, path):
		try:
			with portalocker.Lock(
					path,
					"r",
					flags=portalocker.LOCK_EX,
					timeout=1,
					fail_when_locked=True) as f:
				yield True
		except (portalocker.exceptions.AlreadyLocked, portalocker.exceptions.LockException) as e:
			yield False


class DummyMutex:
	def try_lock(self, processor, path):
		return True

	def unlock(self, processor, path):
		pass

	@contextmanager
	def lock(self, processor, path):
		yield True


if __name__ == "__main__":
	mutex = DatabaseMutex("origami.debug.mutex.db")

	with mutex.lock("proc_a", "/a/b/c") as locked:
		print("try", locked)
		print("retry", mutex.try_lock("proc_a", "/a/b/c"))

	print("clean retry", mutex.try_lock("proc_a", "/a/b/c"))
	mutex.unlock("proc_a", "/a/b/c")
