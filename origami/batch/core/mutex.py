import sqlalchemy
import datetime
import os
import portalocker

from pathlib import Path
from contextlib import contextmanager


class DatabaseMutex:
	def __init__(self, path):
		db_uri = 'sqlite:///%s' % str(Path(path))
		self._engine = sqlalchemy.create_engine(
			db_uri, isolation_level="SERIALIZABLE")

		# see https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#pysqlite-serializable
		@sqlalchemy.event.listens_for(self._engine, "connect")
		def do_connect(dbapi_connection, connection_record):
			# disable pysqlite's emitting of the BEGIN statement entirely.
			# also stops it from emitting COMMIT before any DDL.
			dbapi_connection.isolation_level = None

		@sqlalchemy.event.listens_for(self._engine, "begin")
		def do_begin(conn):
			conn.execute("BEGIN EXCLUSIVE")

		metadata = sqlalchemy.MetaData(self._engine)
		self._mutex_table = sqlalchemy.Table(
			"mutex", metadata,
			sqlalchemy.Column('path', sqlalchemy.Text, nullable=False),
			sqlalchemy.Column('processor', sqlalchemy.Text, nullable=False),
			sqlalchemy.Column('pid', sqlalchemy.BigInteger, nullable=False),
			sqlalchemy.Column('time', sqlalchemy.DateTime, nullable=False),
			sqlalchemy.PrimaryKeyConstraint('path', 'processor', name='mutex_pk')
		)
		metadata.create_all()

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
			stmt = self._mutex_table.delete().where(sqlalchemy.and_(
				self._mutex_table.c.processor == processor,
				self._mutex_table.c.path == path,
				self._mutex_table.c.pid == os.getpid()))
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
