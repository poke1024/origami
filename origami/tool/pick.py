import tkinter as tk
from tkinter import ttk
import tkinter.simpledialog
import click
import sqlite3
import imghdr
import PIL.Image
import PIL.ImageTk
import zipfile
import json
import shapely.wkt
import numpy as np

from pathlib import Path


# https://stackoverflow.com/questions/3352918/how-to-center-a-window-on-the-screen-in-tkinter
def _center_window(win):
	win.update_idletasks()
	width = win.winfo_width()
	height = win.winfo_height()
	x = (win.winfo_screenwidth() // 2) - (width // 2)
	y = (win.winfo_screenheight() // 2) - (height // 2)
	win.geometry('{}x{}+{}+{}'.format(width, height, x, y))


def _annotation_markers(annotations):
	marked = set()
	for line_name, line_text in annotations.items():
		if line_text.strip() == "":
			marked.add(line_name)
	return marked


# from: https://blog.tecladocode.com/tkinter-scrollable-frames/
class ScrollableFrame(ttk.Frame):
	def __init__(self, container, *args, **kwargs):
		super().__init__(container, *args, **kwargs)
		canvas = tk.Canvas(self)
		scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
		self.scrollable_frame = ttk.Frame(canvas)

		self.scrollable_frame.bind(
			"<Configure>",
			lambda e: canvas.configure(
				scrollregion=canvas.bbox("all")
			)
		)

		canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.NW)

		canvas.configure(yscrollcommand=scrollbar.set)

		canvas.pack(side="left", fill="both", expand=True)
		scrollbar.pack(side="right", fill="y")
		self.canvas = canvas
		self.scrollbar = scrollbar


class BetterDialog(tkinter.simpledialog.Dialog):
	def __init__(self, parent, title=None):
		# ignore Dialog constructor.
		tk.Toplevel.__init__(self, parent)

		self.withdraw()  # remain invisible for now
		# If the master is not viewable, don't
		# make the child transient, or else it
		# would be opened withdrawn
		if parent.winfo_viewable():
			self.transient(parent)

		if title:
			self.title(title)

		self.parent = parent

		self.result = None

		body = tk.Frame(self)
		self.initial_focus = self.body(body)
		body.pack(padx=5, pady=5, fill="both", expand=True)

		self.buttonbox()

		if not self.initial_focus:
			self.initial_focus = self

		self.protocol("WM_DELETE_WINDOW", self.cancel)

		self.deiconify()  # become visible now

		self.initial_focus.focus_set()

		# wait for window to appear on screen before calling grab_set
		self.wait_visibility()
		self.grab_set()
		self.wait_window(self)


class GotoPageDialog(BetterDialog):
	def __init__(self, master, app, page_paths, page_index):
		self._app = app
		self._page_paths = page_paths
		self._page_index = page_index
		self.scrollbar = None
		self.listbox = None
		super().__init__(master, title="Goto Page")

	def body(self, master):
		self.geometry("600x400")

		self.scrollbar = tk.Scrollbar(master, orient=tk.VERTICAL)
		self.listbox = tk.Listbox(master, selectmode=tk.SINGLE, yscrollcommand=self.scrollbar.set)
		self.scrollbar.config(command=self.listbox.yview)
		self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		for p in self._page_paths:
			self.listbox.insert("end", str(p))
		self.listbox.see(self._page_index - 1)
		self.listbox.selection_set(self._page_index - 1)

		self.listbox.bind(
			"<<ListboxSelect>>", self._select)
		self.listbox.bind(
			"<Double-Button-1>", self._double_click)

	def _select(self, event):
		now = self.listbox.curselection()

	def _double_click(self, event):
		sel = self.listbox.curselection()
		if sel:
			index = sel[0] + 1
			self._app.goto_page(index)
			self.destroy()

	def buttonbox(self):
		pass


class App:
	def __init__(self, data_path, options):
		self._data_path = Path(data_path)
		self._options = options

		self._active_color = "#FF4500"  # orangered
		self._inactive_color = "#87CEEB"  # skyblue
		self._locked_color = "#FF7F50"  # coral

		page_paths = []
		for p in self._data_path.iterdir():
			if imghdr.what(p) is not None:
				if (p.parent / p.with_suffix(".lines.zip")).exists():
					page_paths.append(p)
		if not page_paths:
			raise click.UsageError(
				"Could not find any line data in %s." % data_path)
		self._page_paths = sorted(page_paths)
		self._page_index = 1

		self._line_items = dict()
		self._line_polys = dict()
		self._annotations = dict()

		db_path = options["db_path"]
		if db_path is None:
			db_path = self._data_path / "annotations.db"
		else:
			db_path = Path(db_path)
		if not db_path.exists():
			raise click.UsageError("%s does not exist." % db_path)
		self._conn = sqlite3.connect(db_path)

		self._build_ui()
		self._load_page()

	def click(self, event):
		canvas = self._frame.canvas
		current = canvas.find_withtag("current")

		if current and current[0] in self._line_items:
			line_name = self._line_items[current[0]]
			if line_name in self._annotations:
				if self._annotations[line_name].strip() == "":
					del self._annotations[line_name]
					canvas.itemconfig("current", outline=self._inactive_color, width=1)
			else:
				self._annotations[line_name] = ""
				canvas.itemconfig("current", outline=self._active_color, width=5)

			canvas.update_idletasks()

	def _build_ui(self):
		self._window = tk.Tk()

		window_size = (1024, 800)
		self._window_width = window_size[0]
		self._window.minsize(*window_size)
		self._window.geometry("%dx%d" % window_size)
		_center_window(self._window)
		self._window.title("origami Line Picking Tool")

		self._page_path_text = tk.StringVar()

		self._upper_labels = tk.Frame(self._window)
		self._page_path_label = tk.Label(
			self._upper_labels, textvariable=self._page_path_text, fg="gray")
		self._page_path_label.pack()
		self._upper_labels.pack(fill="x", padx=8, pady=8)

		self._frame = tk.Frame(self._window, borderwidth=1, bg="gray")
		self._canvas = tk.Canvas(self._frame, bg="gray", bd=0, highlightthickness=0, relief='ridge')
		self._frame.pack(padx=8, pady=8, fill="x")

		# canvas.

		self._frame = ScrollableFrame(self._window)
		self._frame.pack(pady=16, expand=True, fill="x")
		self._frame.canvas.bind("<Button-1>", self.click)

		# bottom frame.

		self._bottom_frame = tk.Frame(self._window, borderwidth=0)

		next_button = tk.Button(self._bottom_frame, text="Goto", command=self.goto_dialog, width=10)
		next_button.pack(padx=8, side="left")
		next_button = tk.Button(self._bottom_frame, text="Next", command=self.next, width=10)
		next_button.pack(padx=8, side="right")
		prev_button = tk.Button(self._bottom_frame, text="Prev", command=self.prev, width=10)
		prev_button.pack(padx=8, side="right")

		self._bottom_frame.pack(side="bottom", fill="x", padx=16, pady=16)

	def goto_dialog(self):
		relative_page_paths = [p.relative_to(self._data_path) for p in self._page_paths]
		GotoPageDialog(self._window, self, relative_page_paths, self._page_index)

	def goto_page(self, index):
		self._save_annotations()
		self._page_index = min(max(1, index), len(self._page_paths))
		self._load_page()

	def _load_page(self):
		page_path = self._page_paths[self._page_index - 1]
		im = PIL.Image.open(page_path)
		original_size = im.size

		relative_page_path = page_path.relative_to(self._data_path)
		self._page_path_text.set("%s [%d/%d]" % (relative_page_path, self._page_index, len(self._page_paths)))

		lines = dict()
		zf_path = page_path.parent / page_path.with_suffix(".lines.zip")
		if zf_path.exists():
			with zipfile.ZipFile(zf_path, "r") as zf:
				for filename in zf.namelist():
					if filename.endswith(".json"):
						line_data = json.loads(zf.read(filename))
						name = filename.rsplit('.', 1)[0]
						lines[name] = shapely.wkt.loads(line_data["wkt"])
		else:
			print("could not load line information for %s" % page_path)

		cursor = self._conn.cursor()
		cursor.execute(
			"SELECT line_path, annotation FROM lines WHERE page_path=?",
			(str(relative_page_path), ))
		self._annotations = dict(list(cursor.fetchall()))
		cursor.close()

		canvas = self._frame.canvas
		max_width = max(960, canvas.winfo_width() - self._frame.scrollbar.winfo_width())
		if im.width > max_width:
			im = im.resize(
				(max_width, int(im.height * (max_width / im.width))),
				resample=PIL.Image.LANCZOS)
		scale = im.size[0] / original_size[0]

		self._frame.scrollable_frame.config(width=im.width, height=im.height)
		canvas.config(width=im.width, height=min(650, im.height))

		self._photo = PIL.ImageTk.PhotoImage(image=im)
		canvas.delete("all")
		canvas.create_image(0, 0, image=self._photo, anchor=tk.NW)

		self._line_items = dict()
		self._line_polys = dict()
		for line_name, line_poly in lines.items():
			coords = np.array(list(line_poly.exterior.coords))
			coords = coords.reshape((coords.size, ))

			text = self._annotations.get(line_name, None)
			is_sel = text is not None
			is_locked = is_sel and text.strip() != ""
			item_spec = canvas.create_polygon(
				*(coords * scale),
				outline=self._locked_color if is_locked else (
					self._active_color if is_sel else self._inactive_color),
				width=5 if is_sel else 1,
				fill='',
				activewidth=5,
				tags=line_name)
			self._line_items[item_spec] = line_name
			self._line_polys[line_name] = line_poly

	def _save_annotations(self):
		marked = _annotation_markers(self._annotations)

		page_path = self._page_paths[self._page_index - 1]
		relative_page_path = page_path.relative_to(self._data_path)

		cursor = self._conn.cursor()
		cursor.execute(
			"SELECT line_path, annotation FROM lines WHERE page_path=?",
			(str(relative_page_path), ))
		old_marked = _annotation_markers(dict(list(cursor.fetchall())))
		cursor.close()

		delta_add = marked - old_marked
		delta_del = old_marked - marked

		if delta_add:
			with self._conn:
				for line_name in delta_add:
					wkt = self._line_polys[line_name].wkt
					row = (str(relative_page_path), line_name, wkt, "", False, False)
					self._conn.execute(
						'''INSERT INTO lines(page_path, line_path, line_wkt, annotation, training, validation)
						VALUES (?, ?, ?, ?, ?, ?)''', row)

		if delta_del:
			with self._conn:
				for line_name in delta_del:
					row = (str(relative_page_path), line_name)
					self._conn.execute(
						"DELETE FROM lines WHERE page_path=? AND line_path=? AND annotation=''", row)

	def next(self):
		self._save_annotations()
		self._page_index += 1
		self._page_index = min(self._page_index, len(self._page_paths))
		self._load_page()

	def prev(self):
		self._save_annotations()
		self._page_index -= 1
		self._page_index = max(self._page_index, 1)
		self._load_page()

	def run(self):
		self._window.mainloop()


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--db-path',
	type=click.Path(exists=True),
	help="Path to db.")
def pick_lines(data_path, **kwargs):
	""" Pick specific lines on document images in DATA_PATH.
	Information from lines batch needs to be present. """
	app = App(data_path, kwargs)
	app.run()


if __name__ == "__main__":
	pick_lines()
