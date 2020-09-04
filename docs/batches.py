batches = [
	{
		"name": "segment",
		"input": ["page.jpg"],
		"output": ["segment.zip"]
	},
	{
		"name": "contours",
		"input": ["segment.jpg"],
		"output": ["contours.0.zip"]
	},
	{
		"name": "flow",
		"input": ["page.jpg", "contours.0.zip"],
		"output": ["flow.zip", "lines.0.zip"]
	},
	{
		"name": "dewarp",
		"input": ["contours.0.zip", "flow.zip"],
		"output": ["contours.1.zip", "dewarp.zip"]
	},
	{
		"name": "layout",
		"input": ["page.jpg", "segment.zip", "contours.0.zip", "lines.0.zip", "contours.1.zip"],
		"output": ["contours.2.zip", "tables.json"]
	},
	{
		"name": "lines",
		"input": ["page.jpg", "segment.zip", "contours.2.zip", "tables.json"],
		"output": ["contours.3.zip", "lines.3.zip"]
	},
	{
		"name": "order",
		"input": ["segment.zip", "contours.1.zip", "contours.2.zip", "contours.3.zip", "lines.3.zip"],
		"output": ["order.json"]
	},
	{
		"name": "ocr",
		"input": ["page.jpg", "lines.3.zip", "tables.json"],
		"output": ["ocr.zip"]
	},
	{
		"name": "compose",
		"input": ["contours.3.zip", "lines.3.zip", "ocr.zip", "order.json", "tables.json"],
		"output": ["compose.zip"]
	}
]

artifacts = []
rows = []
for batch in batches:
	row = dict()
	for inp in batch["input"]:
		if inp not in artifacts:
			artifacts.append(inp)
		row[artifacts.index(inp)] = "in"
	for out in batch["output"]:
		if out not in artifacts:
			artifacts.append(out)
		row[artifacts.index(out)] = "out"
	rows.append(row)

# see https://codepen.io/chriscoyier/pen/Fapif
css = """
.table-header-rotated {
  border-collapse: collapse;
}

.table-header-rotated td {
  width: 30px;
  text-align: center;
  padding: 10px 5px;
  border: 1px solid #ccc;
}

th.rotate {
  /* Something you can count on */
  height: 140px;
  white-space: nowrap;
}

th.rotate > div {
  transform: 
    /* Magic Numbers */
    translate(25px, 51px)
    /* 45 is really 360 - 45 */
    rotate(315deg);
  width: 30px;
}
th.rotate > div > span {
  border-bottom: 1px solid #ccc;
  padding: 5px 10px;
}

th.row-header {
  padding: 0 10px;
  border-bottom: 1px solid #ccc;
}
}"""

html = []

html.append("<!DOCTYPE html>")
html.append("<hmtl><body>")
html.append("<style>%s</style>" % css)

html.append('<table class="table table-header-rotated">')

html.append("<thead>")
html.append("<tr>")

html.append("<th></th>")
for a in artifacts:
	html.append('<th class="rotate"><div><span>%s</span></div></th>' % a)

html.append("</tr>")
html.append("</thead>")

html.append("<tbody>")
for batch, row in zip(batches, rows):
	html.append("<tr>")
	html.append('<th class="row-header">%s</th>' % batch["name"])

	for i, a in enumerate(artifacts):
		code = ""
		if row.get(i) == "in":
			code = "&#9711;"
		if row.get(i) == "out":
			code = "&#11044;"
		html.append("<td>%s</td>" % code)

	html.append("</tr>")
html.append("</tbody>")

html.append("</table>")

html.append("</body></hmtl>")


with open("batches.html", "w") as f:
	f.write("\n".join(html))
