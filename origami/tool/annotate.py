import tkinter as tk
import tkinter.simpledialog as simpledialog
import tkinter.messagebox as messagebox
import tkinter.font
import PIL.Image
import PIL.ImageTk
import click
import sqlite3
import re
import pandas as pd
import collections
import numpy as np

from pathlib import Path
from origami.tool.lineload import LineLoader


# https://stackoverflow.com/questions/3352918/how-to-center-a-window-on-the-screen-in-tkinter
def _center_window(win):
	win.update_idletasks()
	width = win.winfo_width()
	height = win.winfo_height()
	x = (win.winfo_screenwidth() // 2) - (width // 2)
	y = (win.winfo_screenheight() // 2) - (height // 2)
	win.geometry('{}x{}+{}+{}'.format(width, height, x, y))


def _gt_txt_path_to_line_path(path):
	name = path.split("/")[-1]
	name = re.sub(r"\.gt\.txt$", "", name)
	parts = name.split("-")
	# last 4 parts are: classifier, label, block id, line id
	line_path = "/".join(parts[-4:])
	page_path = "-".join(parts[:-4])
	page_path = re.sub(r"-jpg$", ".jpg", page_path)
	return page_path, line_path


class Navigator(tk.Frame):
	def __init__(self, app):
		super().__init__(app.window, borderwidth=0)

		self._app = app
		self._conn = app.connection
		self._format_row = None

		cursor = self._conn.cursor()
		cursor.execute("""
			SELECT page_path, line_path, annotation, author FROM lines
			ORDER BY page_path, line_path""")
		self._master = [tuple([x or "" for x in row]) for row in cursor.fetchall()]
		cursor.close()

		head = tk.Frame(self, borderwidth=0)

		self._nav_search_var = tk.StringVar()
		self._nav_search_var.trace_add("write", lambda vname, vindex, op: self._search())
		self._nav_search = tk.Entry(head, width=40, textvariable=self._nav_search_var)

		self._nav_search.pack(padx=0, pady=4, fill="x", side="left")

		self._regex_var = tk.BooleanVar()
		self._case_var = tk.BooleanVar()
		self._meta_var = tk.BooleanVar()
		self._meta_var.set(True)

		for name, var in (("regex", self._regex_var), ("case", self._case_var), ("meta", self._meta_var)):
			button = tk.Checkbutton(
				head, text=name, command=lambda: self._search(), variable=var)
			button.pack(padx=8, pady=4, side="right")

		self._head = head

		list_area = tk.Frame(self, borderwidth=0)

		self._nav_scrollbar = tk.Scrollbar(list_area, orient=tk.VERTICAL)
		font = tkinter.font.Font(family=self._app.font_name, size=14)
		self._nav_listbox = tk.Listbox(
			list_area, selectmode=tk.SINGLE, yscrollcommand=self._nav_scrollbar.set, font=font)
		self._nav_scrollbar.config(command=self._nav_listbox.yview)
		self._nav_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		self._nav_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		self._fill_nav_listbox()
		self._nav_listbox.selection_set(0)
		self._nav_listbox.bind("<<ListboxSelect>>", self.navigate_to)
		#self._nav_listbox.bind("<Double-Button-1>", self.navigate_to)

		self._list_area = list_area

		self._bottom_frame = tk.Frame(self, borderwidth=0)

		toggle_list_button = tk.Button(self._bottom_frame, text="List", command=self.toggle_list, width=10)
		self._toggle_list_button = toggle_list_button
		toggle_list_button.pack(padx=8, side="left")

		next_button = tk.Button(self._bottom_frame, text="Next", command=self.next, width=10)
		next_button.pack(padx=8, side="right")
		prev_button = tk.Button(self._bottom_frame, text="Prev", command=self.prev, width=10)
		prev_button.pack(padx=8, side="right")

		self._bottom_frame.pack(side="bottom", fill="x", padx=16, pady=16)

	def toggle_list(self):
		if self._head.winfo_ismapped():
			self._head.pack_forget()
			self._list_area.pack_forget()
		else:
			self._head.pack(fill="x")
			self._list_area.pack(fill="both", padx=4)

	def is_searching(self):
		return self._nav_search_var.get().strip() != ""

	def _find_next(self, rows):
		text = self._nav_search_var.get()

		search_meta = self._meta_var.get()
		is_case_sensitive = self._case_var.get()

		if not self._regex_var.get():
			text = re.escape(text)

		try:
			pattern = re.compile(
				text, 0 if is_case_sensitive else re.IGNORECASE)
		except re.error:
			self.bell()
			return None

		for i, (page_path, line_path, annotation, author) in enumerate(rows):
			if search_meta:
				subject = "|".join([page_path, line_path, annotation, author])
			else:
				subject = annotation
			if pattern.search(subject):
				return i

		return None

	def _search(self):
		i = self._find_next(self._master)
		if i is not None:
			self.select(i)
		else:
			self._nav_listbox.select_clear(0, "end")

	def select(self, index):
		self._nav_listbox.select_clear(0, "end")
		self._nav_listbox.selection_set(index)
		self._nav_listbox.see(index)
		self._nav_listbox.activate(index)
		self._nav_listbox.selection_anchor(index)
		self.navigate_to(None)

	def first(self):
		self.select(0)

	def last(self):
		self.select(len(self._master) - 1)

	def ui_goto_index(self):
		line_index = simpledialog.askinteger("Goto", "Enter Line Index:")
		if line_index:
			if 1 <= line_index <= len(self._master):
				self.select(line_index - 1)
			else:
				self.bell()

	def next(self):
		sel = self._nav_listbox.curselection()
		if sel and len(sel) == 1:
			index = sel[0]
			if index + 1 < len(self._master):
				if self.is_searching():
					i = self._find_next(self._master[index + 1:])
					if i is not None:
						self.select(index + 1 + i)
					else:
						self.bell()
				else:
					self.select(index + 1)
			else:
				self.bell()
		else:
			self.bell()

	def prev(self):
		sel = self._nav_listbox.curselection()
		if sel and len(sel) == 1:
			index = sel[0]
			if index - 1 >= 0:
				if self.is_searching():
					i = self._find_next(reversed(self._master[:index]))
					if i is not None:
						self.select(index - 1 - i)
					else:
						self.bell()
				else:
					self.select(index - 1)
			else:
				self.bell()
		else:
			self.bell()

	@property
	def line_cursor(self):
		sel = self._nav_listbox.curselection()
		if sel:
			index = sel[0]
			return self._master[index][:2]
		else:
			return None

	def _reorder(self, new_order):
		cursor = self.line_cursor
		self._master = new_order
		index = [tuple(x[:2]) for x in self._master].index(cursor)

		self._fill_nav_listbox()
		self.select(index)

	def sort_by_name(self):
		self._reorder(sorted(self._master, key=lambda row: row[:2]))

	def sort_by_length(self):
		self._reorder(sorted(self._master, key=lambda row: len(row[2])))

	def _fill_nav_listbox(self):
		rows = self._master

		enum_rows = [(str(i + 1), *args) for i, args in enumerate(rows)]

		lens = np.array([[len(x) for x in row] for row in enum_rows], dtype=np.int16)
		max_lens = np.max(lens, axis=0)

		def format_row(row):
			cols = ["%-*s" % (int(max_len), s) for max_len, s in zip(max_lens, row)]
			return " | ".join(cols)
		self._format_row = format_row

		self._nav_listbox.delete(0, "end")

		for row in enum_rows:
			self._nav_listbox.insert("end", format_row(row))

	def navigate_to(self, event):
		cursor = self.line_cursor
		if cursor is not None:
			self._app.goto(*cursor)

	def update_annotation(self, page_path, line_path, annotation, author):
		index = [tuple(x[:2]) for x in self._master].index((page_path, line_path))
		self._master[index] = (page_path, line_path, annotation, author)
		if self._format_row is None:
			self._fill_nav_listbox()
			self._nav_listbox.see(index)
		else:
			self._nav_listbox.delete(index)
			self._nav_listbox.insert(index, self._format_row(
				(index, page_path, line_path, annotation, author)))


class ReviewIterator:
	def __init__(self, xlsx_path):
		dfs = pd.read_excel(xlsx_path, sheet_name=None)
		df = dfs["evaluation - per line"]
		self._df = df.sort_values(by="ERR", ascending=False)
		self._index = 0

	def goto_next(self):
		self._index = min(self._index + 1, len(self._df) - 1)

	def goto_prev(self):
		self._index = max(self._index - 1, 0)

	@property
	def prediction(self):
		return self._df["PRED"][self._index]

	@property
	def err(self):
		return self._df["ERR"][self._index]

	@property
	def cer(self):
		return self._df["CER"][self._index]

	@property
	def line_path(self):
		return _gt_txt_path_to_line_path(self._df["GT FILE"][self._index])


Annotation = collections.namedtuple("Annotation", ["text", "author", "training", "validation"])


class App:
	def __init__(self, data_path, options):
		self._data_path = Path(data_path)
		self._options = options

		self._font_name = "Andale Mono"
		self._author = options["author"]
		if self._author == "Anonymous":
			options["read_only"] = True

		db_path = options["db_path"]
		if db_path is None:
			db_path = self._data_path / "annotations.db"
		else:
			db_path = Path(db_path)
		if not db_path.exists():
			raise click.UsageError("%s does not exist." % db_path)
		self._conn = sqlite3.connect(db_path)

		self._line_loader = LineLoader()
		self._occurrences = []
		self._current_occurrence = 0

		if options["review"]:
			self._review = ReviewIterator(options["review"])
		else:
			self._review = None

		self._build_ui()
		self._line_cursor = self._navigator.line_cursor
		self._line_image = None

		if self._review:
			self._sync_review()

		self._update_image()

		self._window.bind("<Key>", self._key)

	@property
	def window(self):
		return self._window

	@property
	def connection(self):
		return self._conn

	@property
	def font_name(self):
		return self._font_name

	def _sync_review(self):
		index = self.line_index(*self._review.line_path)
		if index is None:
			# index might stem from another annotation file.
			raise RuntimeError("did not found", self._review.line_path)
		self._review_pred_var.set(self._review.prediction)
		self._review_cer_var.set("CER: %.2f%%" % (self._review.cer * 100))
		self._review_err_var.set("ERR: %s" % self._review.err)

	def click_line_path_label(self, event):
		self._window.clipboard_clear()
		self._window.clipboard_append(self._line_path_text.get())

	def click_canvas(self, event):
		pass

	def _build_ui(self):
		self._window = tk.Tk()

		# does not work on macOS:
		# https://stackoverflow.com/questions/1800452/how-to-intercept-wm-delete-window-on-osx-using-tkinter
		self._window.protocol("WM_DELETE_WINDOW", self._save_annotation)

		window_size = (1024, 400)
		self._window_width = window_size[0]
		self._window.minsize(*window_size)
		self._window.geometry("%dx%d" % window_size)
		_center_window(self._window)
		self._window.title("origami Annotation Tool")

		self._line_path_text = tk.StringVar()

		self._upper_labels = tk.Frame(self._window)
		self._line_path_label = tk.Label(
			self._upper_labels, textvariable=self._line_path_text, fg="gray")
		self._line_path_label.pack(side="left")
		self._line_path_label.bind("<Button-1>", self.click_line_path_label)

		self._edit_mode_var = tk.IntVar()
		edit_button = tk.Checkbutton(
			self._upper_labels, text="Edit", command=self.toggle_edit_mode,
			variable=self._edit_mode_var)
		if not self._options["read_only"]:
			edit_button.pack(padx=8, side="right")
		self._edit_button = edit_button

		self._current_author_label_text = tk.StringVar()
		self._current_author_label_text.set(self._author)
		self._current_author_label = tk.Label(
			self._upper_labels, textvariable=self._current_author_label_text, fg="gray", justify="right")
		self._current_author_label.pack(side="right")

		self._upper_labels.pack(fill="x", padx=8, pady=8)

		self._frame = tk.Frame(self._window, borderwidth=1, bg="gray")
		self._canvas = tk.Canvas(self._frame, bg="gray", bd=0, highlightthickness=0, relief='ridge')
		self._frame.pack(padx=8, pady=8, fill="x")
		self._canvas.bind("<Button-1>", self.click_canvas)

		# review frame.

		self._review_frame = tk.LabelFrame(self._window, text="Review")

		if self._review:
			font14 = tkinter.font.Font(family=self._font_name, size=14)
			self._review_pred_var = tk.StringVar()
			self._review_pred = tk.Label(
				self._review_frame, textvariable=self._review_pred_var, font=font14)
			self._review_pred.pack(side="top", fill="x")

			font10 = tkinter.font.Font(family=self._font_name, size=10)
			self._review_cer_var = tk.StringVar()
			self._review_cer = tk.Label(
				self._review_frame, textvariable=self._review_cer_var, font=font10, justify="right")
			self._review_cer.pack(side="right")

			self._review_err_var = tk.StringVar()
			self._review_err = tk.Label(
				self._review_frame, textvariable=self._review_err_var, font=font10, justify="right")
			self._review_err.pack(side="right")

			self._review_frame.pack(fill="x", padx=16, pady=16)

		# annotation frame.

		self._ann_frame = tk.Frame(self._window, borderwidth=0)

		annotation_font = tkinter.font.Font(family=self._font_name, size=20)
		self._annotation_entry_var = tk.StringVar()
		self._annotation_entry = tk.Entry(
			self._ann_frame, textvariable=self._annotation_entry_var, font=annotation_font)
		self._annotation_entry.pack(padx=8, pady=8, fill="x")
		self._annotation_entry.config(state="disabled")

		self._ann_frame.pack(fill="x")

		# edit frame

		self._edit_frame = tk.Frame(self._window, borderwidth=0)

		self._sets_vars = dict(training=tk.IntVar(), validation=tk.IntVar())
		self._sets_cb = dict()
		for s in ("Training", "Validation"):
			self._sets_cb[s.lower()] = tk.Checkbutton(
				self._edit_frame, text=s,
				variable=self._sets_vars[s.lower()])

		self._last_author_label_text = None
		'''
		self._last_author_label_text = tk.StringVar()
		self._last_author_label_label = tk.Label(
			self._edit_frame, textvariable=self._last_author_label_text, fg="gray", justify="right")
		self._last_author_label_label.pack(side="right", padx=8)
		'''

		undo_button = tk.Button(self._edit_frame, text="Undo", command=self.undo, width=10)
		self._undo_button = undo_button

		self._edit_frame.pack(fill="x")

		# navigation frame.

		self._navigator = Navigator(self)
		self._navigator.pack(side="bottom", fill="x", pady=16, padx=8)

		# menu

		menubar = tk.Menu(self._window, tearoff=0)
		self._window.config(menu=menubar)

		gotoMenu = tk.Menu(menubar, tearoff=0)
		gotoMenu.add_command(label="Next", command=self._navigator.next)
		gotoMenu.add_command(label="Previous", command=self._navigator.prev)
		gotoMenu.add_separator()
		gotoMenu.add_command(label="First", command=self._navigator.first)
		gotoMenu.add_command(label="Last", command=self._navigator.last)
		gotoMenu.add_command(label="Goto Index…", command=self._navigator.ui_goto_index)
		gotoMenu.add_separator()
		gotoMenu.add_command(label="By Name", command=self._navigator.sort_by_name)
		gotoMenu.add_command(label="By Length", command=self._navigator.sort_by_length)

		menubar.add_cascade(label="Navigation", menu=gotoMenu)

	def run(self):
		self._window.mainloop()

	def toggle_edit_mode(self):
		if self.is_in_edit_mode():
			self._undo_button.pack(padx=8, side="left")
			for _, cb in self._sets_cb.items():
				cb.pack(padx=8, side="right")
			self._annotation_entry.config(state="normal")
		else:
			self._save_annotation(force_save=True)
			self._undo_button.pack_forget()
			for _, cb in self._sets_cb.items():
				cb.pack_forget()
			self._annotation_entry.config(state="disabled")

	def is_in_edit_mode(self):
		if self._options["read_only"]:
			return False
		else:
			return self._edit_mode_var.get() != 0

	def undo(self):
		self._update_image()

	def goto(self, page_path, line_path):
		new_cursor = (page_path, line_path)
		if new_cursor != self._line_cursor:
			self._save_annotation()
			self._line_cursor = new_cursor
			self._update_image()

	def label(self, label):
		path = self._image_paths[self._image_index - 1]
		with open(path.parent / (path.stem + ".label.txt"), "w") as f:
			f.write(label)

	def _update_image(self):
		page_path, line_path = self._line_cursor

		self._line_path_text.set("Loading Page...")
		#self._author_label_text.set("")
		self._window.update()

		line_height = 64
		im = self._line_loader.load_line_image(
			self._data_path / page_path,
			line_path,
			target_height=line_height,
			deskewed=True,
			binarized=False)
		self._line_image = im

		self._line_path_text.set("%s/%s" % (
			page_path.rsplit(".", 1)[0], line_path))

		max_image_width = self._window_width - 32
		if im.width > max_image_width:
			im.thumbnail((max_image_width, line_height), resample=PIL.Image.LANCZOS)

		width, height = im.size

		self._frame.config(width=width, height=line_height)
		self._canvas.config(width=width, height=line_height)

		self._frame.pack(padx=8, pady=8, fill="x")
		self._canvas.pack(anchor="center")

		self._photo = PIL.ImageTk.PhotoImage(image=im)
		self._canvas.delete("all")
		self._canvas.create_image(0, line_height / 2 - height / 2, image=self._photo, anchor=tk.NW)

		ann = self._load_annotation(page_path, line_path)
		self._annotation_entry_var.set(ann.text)
		self._sets_vars["training"].set(ann.training)
		self._sets_vars["validation"].set(ann.validation)

		if self._last_author_label_text:
			if ann.author.strip():
				self._last_author_label_text.set("%s" % ann.author)
			else:
				self._last_author_label_text.set("")

	def _load_annotation(self, page_path, line_path):
		cursor = self._conn.cursor()
		cursor.execute(
			"SELECT annotation, author, training, validation FROM lines WHERE page_path=? AND line_path=?",
			(page_path, line_path))
		row = cursor.fetchone()
		cursor.close()
		return Annotation(text=row[0], author=row[1], training=row[2], validation=row[3])

	def _save_annotation(self, force_save=False):
		if not self.is_in_edit_mode() and not force_save:
			return

		if self._options["read_only"]:
			return

		page_path, line_path = self._line_cursor
		old_annotation = self._load_annotation(page_path, line_path)

		new_annotation = Annotation(
			text=self._annotation_entry_var.get(),
			author=old_annotation.author,
			training=self._sets_vars["training"].get(),
			validation=self._sets_vars["validation"].get())
		if old_annotation != new_annotation:
			with self._conn:
				self._conn.execute(
					"UPDATE lines SET annotation=?, author=?, training=?, validation=? WHERE page_path=? AND line_path=?", (
					new_annotation.text,
					self._author,
					new_annotation.training,
					new_annotation.validation,
					page_path, line_path))

			self._navigator.update_annotation(page_path, line_path, new_annotation.text, self._author)

	def _key(self, event):
		is_command_key = (event.state & 0x08) != 0

		if event.keysym == 'Right' and is_command_key:
			self._navigator.next()
		if event.keysym == 'Left' and is_command_key:
			self._navigator.prev()


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--db-path',
	type=click.Path(exists=True),
	help="Path to db.")
@click.option(
	'--read-only',
	is_flag=True,
	help="Open in read-only mode.")
@click.option(
	'--author',
	type=str,
	default="Anonymous",
	help="Author name to use in edit mode.")
@click.option(
	'--review',
	type=click.File(mode="rb"),
	help="Path to xlsx file for review.")
def annotate_lines(data_path, **kwargs):
	""" Work on annotation database and document images in DATA_PATH.
	Information from lines batch needs to be present. """
	app = App(data_path, kwargs)
	app.run()


if __name__ == "__main__":
	annotate_lines()
