# req:
# paramiko awesome-slugify

import paramiko
import stat
import re
import sqlite3
import tqdm
import contextlib
import shutil
import os
import sys
import logging
import traceback
import tempfile
import subprocess
import hashlib

from pathlib import Path
from slugify import slugify


def get_digest(file_path):
	# https://stackoverflow.com/questions/22058048/hashing-a-file-in-python

	h = hashlib.sha256()

	with open(file_path, 'rb') as file:
		while True:
			# Reading is buffered, so we can read smaller chunks.
			chunk = file.read(h.block_size)
			if not chunk:
				break
			h.update(chunk)

	return h.hexdigest()


def get_signature(path):
	stat = path.stat()
	return (stat.st_mtime, stat.st_size, get_digest(path))


class WorkingSet:
	def __init__(self, sftp, local_path, remote_path):
		self._sftp = sftp
		self._local_path = local_path
		self._remote_path = remote_path
		self._signatures = {}

	def _copy_get(self, src, dst):
		fileattr = self._sftp.lstat(src)
		is_dir = stat.S_ISDIR(fileattr.st_mode)
		if is_dir:
			dst.mkdir()
			for p in sorted(self._sftp.listdir(src)):
				self._copy_get(f"{src}/{p}", dst / p)
		else:
			self._sftp.get(src, dst)
			self._signatures[dst] = get_signature(dst)

	def _copy_put(self, local_path, remote_path):
		for p in local_path.iterdir():
			if p.is_dir():
				self._copy_put(p, f"{remote_path}/{p.name}")
			else:
				sig = self._signatures.get(p)
				if sig != get_signature(p):
					self._sftp.put(p, f"{remote_path}/{p.name}")

	def add(self, remote_name):
		self._copy_get(
			f"{self._remote_path}/{remote_name}",
			self._local_path / remote_name)

	def sync(self):
		self._copy_put(self._local_path, self._remote_path)


class Scheduler:
	def __init__(self, sftp, con, config, isolated_work_path):
		self._sftp = sftp
		self._con = con

		with con:
			con.execute('''
				CREATE TABLE IF NOT EXISTS task(path text primary key, done int)''')
			con.execute('''
				CREATE TABLE IF NOT EXISTS meta(key text, value text)''')

		self._work_path_root = isolated_work_path
		self._work_path_root.mkdir(exist_ok=True)

		self._remote_root_path = config['connection']['remote_root_path']

		meta = {}
		for k, v in self._con.execute("SELECT * FROM meta").fetchall():
			meta[k] = v
		if self.num_tasks != 0:
			stored_remote_root_path = meta.get("remote_root_path")
			if self._remote_root_path != stored_remote_root_path:
				raise ValueError(f"stored remote root path is {stored_remote_root_path}, expected {self._remote_root_path}")

	def classify_node(self, path):
		raise NotImplementedError()

	def prepare_working_set(self, working_set, filename):
		raise NotImplementedError()

	def run_task(self, local_path):
		raise NotImplementedError()

	def copy_get(self, src, dst):
		parts = src.split("/")
		filename = parts[-1]

		fileattr = self._sftp.lstat(src)
		is_dir = stat.S_ISDIR(fileattr.st_mode)
		if is_dir:
			local_dir = dst / filename
			local_dir.mkdir()
			for p in sorted(self._sftp.listdir(src)):
				self.copy_get(f"{src}/{p}", local_dir)
		else:
			self._sftp.get(src, dst / filename)

	def copy_put(self, src, dst):
		raise NotImplementedError()

	@contextlib.contextmanager
	def run_task_context(self, remote_path):
		work_path = Path(tempfile.mkdtemp(dir=self._work_path_root))

		try:
			remote_path_base, remote_path_filename = remote_path.rsplit("/", 1)
			working_set = WorkingSet(self._sftp, work_path, remote_path_base)

			self.prepare_working_set(working_set, remote_path_filename)

			# perform actual computations on local data.
			yield work_path

			working_set.sync()

		finally:
			shutil.rmtree(work_path)

	@property
	def num_tasks(self):
		return self._con.execute("SELECT COUNT(*) FROM task").fetchone()[0]

	def add_tasks(self):
		con = self._con

		with tqdm.tqdm() as pbar:

			with con:

				def collect_tasks(remote_path):
					pbar.set_description(remote_path)
					pbar.refresh()
					pbar.update(1)

					for p in sorted(self._sftp.listdir(remote_path)):
						full_path = f'{remote_path}/{p}'
						c = self.classify_node(full_path)
						if c == "ignore":
							continue
						elif c == "key":
							con.execute("insert into task(path, done) values (?, ?)", (full_path, 0))
						elif c == "descend":
							fileattr = self._sftp.lstat(full_path)
							is_dir = stat.S_ISDIR(fileattr.st_mode)
							if is_dir:
								collect_tasks(full_path)
						else:
							raise ValueError(c)

				collect_tasks(self._remote_root_path)
				con.execute(
					"insert into meta(key, value) values(?, ?)",
					("remote_root_path", self._remote_root_path))

	def run_tasks(self):
		con = self._con

		for p in con.execute('SELECT path from task WHERE done=0').fetchall():
			try:
				remote_path = p[0]
				logging.info(f"working on {remote_path}")
				with self.run_task_context(remote_path) as local_path:
					self.run_task(local_path)
				with con:
					con.execute('UPDATE task SET done=1 WHERE path=?', (remote_path,))
			except KeyboardInterrupt:
				raise
			except SystemExit:
				raise
			except:
				logging.error(f"failed to process {p}")
				traceback.print_exc()


class OrigamiScheduler(Scheduler):
	def __init__(self, sftp, con, config, isolated_work_path, processors):
		super().__init__(sftp, con, config, isolated_work_path)

		self._pattern = re.compile(r"/jpg/.*\.jpg$")
		self._processors = processors

	def classify_node(self, path):
		if path.endswith(".out"):
			return "ignore"

		if self._pattern.search(path):
			return "key"

		parts = path.split("/")

		if parts[-1] == "presentation":
			return "ignore"
		else:
			return "descend"

	def prepare_working_set(self, working_set, filename):
		working_set.add(filename)
		filename_base = filename.rsplit(".", 1)[0]
		working_set.add(filename_base + ".out")

	def run_task(self, local_path):
		for processor in self._processors:
			logging.info(f"running {processor.processor_name}")
			processor.traverse(local_path)


def run_on_remote_data(config, processors):
	connection_config = config["connection"]

	pkey = paramiko.RSAKey.from_private_key_file(
		connection_config['client_private_key_file'])

	transport = paramiko.Transport((connection_config['host'], connection_config['port']))
	transport.connect(username=connection_config['username'], pkey=pkey)
	sftp = paramiko.SFTPClient.from_transport(transport)

	try:
		connection_key = "-".join([slugify(str(x)) for x in [
			connection_config["unique_id"],
			connection_config["host"],
			connection_config["port"],
			connection_config["remote_root_path"]]])

		work_path = Path(config['client_work_data_path'])
		isolated_work_path = work_path / connection_key
		isolated_work_path.mkdir(exist_ok=True)

		db_path = isolated_work_path / "tasks.sqlite3"

		con = sqlite3.connect(db_path)
		try:
			scheduler = OrigamiScheduler(
				sftp, con, config,
				isolated_work_path,
				processors)

			if scheduler.num_tasks == 0:
				scheduler.add_tasks()

			print(f"found {scheduler.num_tasks} tasks.")

			scheduler.run_tasks()

		finally:
			con.close()

	finally:
		sftp.close()
