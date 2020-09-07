# Origami Artifact File Formats

- [segment.zip](#segmentzip)
- [contours.zip](#contourszip)
- [flow.zip](#flowzip)
- [lines.zip](#lineszip)
- [dewarp.zip](#dewarpzip)
- [tables.json](#tablesjson)
- [order.json](#orderjson)
- [ocr.zip](#ocrzip)
- [compose.zip](#composezip)

## page image

The original unmodified image file of the document, typically a `.png` or `.jpg`
file.

## segment.zip

Contains pixel-wise predictions for the document (i.e. contains the output from
the DNN segmentation stage). There can be various different prediction types (e.g.
regions and separators).

For each prediction type, there is a png file containing the prediction
labels (as color indices using a paletted png) and an accompanying json file.
Note that the size of the pngs need not match the document image size. Origami
will rescale predictions to the document resolution automatically.

Example of the file contents for two prediction types "regions" and "separators":

```
regions.json
regions.png
separators.json
separators.png
```

The JSON files are structured like this:

```
interface Prediction {
    type: "SEPARATOR" | "REGION";
    name: string;
    classes: { [labelName: string]: int };
}
```

The `REGION` type is meant for predictions that define regions of interest such as text or illustrations; the underlying shapes are polygons. The `SEPARATOR` is meant for predictions that describe lines or curves that delinate different areas; the underlying shapes are polylines.

`name` gives a name to this kind of specific prediction that is used throughout later stages of the pipeline to refer to elements derived from it - we call this predictor name or simply pname. `classes` maps the label indices in the png to meaningful names that are used in subsequent pipeline stages to refer to the labels.

Example:

```
{"type": "REGION",
"name": "regions",
"classes": {"TEXT": 0, "TABULAR": 1, "ILLUSTRATION": 2, "BACKGROUND": 3}}
```

## contours.zip

Contains polygonized region and separator predictions originally extracted from `segment.zip`. The file structure is as follows:

```
meta.json
pname1/
	label1/
		0.wkt
		1.wkt
		...
	label2/
		0.wkt
		1.wkt
		...
	...
pname2/
	...
...
```

`meta.json` contains information about the underlying predictor type for each
predictor name (see `ContoursMeta` below).

```
interface ContoursMeta {
    version: number; // currently 2
    predictions: Array<ContoursPrediction>;
}

interface ContoursPrediction {
    name: string;
    type: "SEPARATOR" | "REGION";
}
```

This is a partial replication of the data in `segment.zip`. Example of a `meta.json` file:

```
{"version": 2, "predictions": [
	{"name": "separators", "type": "SEPARATOR"},
	{"name": "regions", "type": "REGION"}]}
```

For each predictor (e.g. "regions") and label (e.g. "TEXT") there is a folder
containing all polygonal shapes of that combination (the names, e.g. `0.wkt`,
are sequentially indexed and start with 0). Each `.wkt` file contains an
OpenGIS Well Known Text (WKT) string describing the shape. This form is expected
to be a single polygon.

This tells Origami that the predictor called "regions" is actually a `REGION` predictor that predicts regions. The file format supports several `REGION` predictors with different names, although Origami's implementation is currently not fully prepared to handle this.

For `SEPARATOR` predictors an additional `meta.json` file is located at the level of the
`wkt` files, which contains - for each `wkt` file - the estimated average line width
of the separator line in pixels (these numbers relate to the original pixel-wise
prediction png and are not scaled to the image document size).  Example:

```
{"width": [1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
```

## contours.0.zip

Stage-dependent instantiation of [contours.zip](#contourszip). Describes contours on
the original *warped* page.

## contours.1.zip

Stage-dependent instantiation of [contours.zip](#contourszip). Describes contours on
the *dewarped* page.

## contours.2.zip

Stage-dependent instantiation of [contours.zip](#contourszip). Describes *aggregate*
contours on the *dewarped* page, which fix over- and under-segmentation issues.

For tabular regions, this step introduces a modified naming scheme that divides
those regions into sub regions, based on horizontal divisions and rows. 

## contours.3.zip

Stage-dependent instantiation of [contours.zip](#contourszip). Describes *aggregate*,
*reliable* contours on the *dewarped* page, which are contours refined with
results from line detection.

## flow.zip

Contains information about page flow/warping in the form of a set of locations and
angles. We differentiate two kinds of flow information: horizontal (h) and vertical
(v).

Horizontal flow shows the deviation of a line we send over the page from left to right.
Vertical flow flow shows the deviation of a line we send over the page from top to bottom.

For each kind of flow, we give samples of this information in the form of points
and accompanying angles. In this way, we for example specify that a top-to-bottom
line running through a certain point will - at that point - have the given slope 
or angle.

The data is stored in four files inside the zip file, two for each kind of flow:

```
h.npy
h.json
v.npy
v.json
```

The `npy` files are numpy arrays. They contain flow samples as 3-tuples
`(x, y, phi)`. One such tuple specifies that at point `(x, y)` we see angle `phi`.
An array's numpy shape is `(n, 3)` when n is the number of samples. Note that the
location of the samples is not bound to form any sort of grid or regular structure.

The `json` file contains a `version` and the overall size of the
page, i.e. the range in which the sampled points lie:

```
interface FlowMeta {
    version: number; // currently 1
    size: [number, number]; // width, height
}
```

Example:

```
{"version": 1, "size": [2400, 3431]}
```

## lines.zip

Contains results of line detection, i.e. detailed information on the geometry of
lines that is ordered by regions.

For each detected line, the zip file contains one JSON file that is located inside
the line's region path. Here is an overall example structure of the zip file:

Inside the zip file, line JSONs are contained within the path of the region the line
belongs to. The structure, ordering and indexing directly corresponds to the region
paths found in [contours.zip](#contourszip).

Here is an example:

```
meta.json
regions/
    TABULAR/
        ...
    TEXT/
        0/
            0.json
            1.json
            ...
        ...
    ...
```

In the example above, `0.json` describes the first line that belongs to
the first region (id `0`) of the `TEXT` predictor. Internally, Origami
will refer to this line as `("regions", "TEXT", 0, 0)`.

`meta.json` currently only contains a version id, i.e. `{"version": 1}`.

The contents of each line JSON are structured as given in `Line`:

```
interface Line {
    p: [number, number]; // x, y
    right: [number, number]; // x, y
    up: [number, number]; // x, y
    wkt: string;
    confidence: number; // between 0 and 1
    tesseract_data: DetectionData;
}

interface DetectionData {
    baseline: [[number, number], [number, number]]; // left and right points
    descent: number:
    ascent: number:
    height: number:
}
```

Most of the time, Origami regards lines as rectangular forms (they may be rotated).

`p` is the bottom left location of this line rectangle, `up` and `right` are vectors
starting from there describing the top left and right bottom points of line rectangle.

`wkt` gives a detailed form of the line (as OpenGIS Well Known Text), which is expected
to be a single polygon. This form might be more complex than the full rectangle due to the
complexity of  the shape of the region that contains the line.

`confidence` gives an estimate of how reliable this line is with regards to region that
it is contained in. Set this to 1 if it is fully reliable, and to 0 if the geometry is
erroneous and should be skipped. Origami calculates this value from pixelwise segmentation
data.

`tesseract_data` gives more detailed data retrieved during line detection, such as
the original baseline position as well as information on font height, descent and ascent.
Origami tries to extend baselines to region boundaries, therefore the original baseline
might be shorter than the rectangle described through `p` and `right`.

## lines.0.zip

Stage-dependent instantiation of [lines.zip](#lineszip). Describes lines on the original
*warped* page.

## lines.3.zip

Stage-dependent instantiation of [lines.zip](#lineszip). Describes reliable lines on a
*dewarped* page.

## dewarp.zip

Contains a grid that allows an easy transformation of the warped page into a dewarped
page. This zip file contains two files:

```
meta.json
data.npy
```

`meta.json` contains the following information: 

```
interface DewarpMeta {
    version: number; // currently 1
    cell: number; // pixel size (width and height) of grid cells
    shape: [number, number, number]; // shape of grid
}
```

Example:

```
{"version": 1, "cell": 25, "shape": [140, 99, 2]}
```

Through `cell` and `shape`, this describes a regularly spaced grid
on the dewarped page.

`shape` is also the numpy shape of the numpy array in `data.npy`.
The latter contains, for each grid point on the dewarped page, a
corresponding location on the original warped page.

Mapping from warped to dewarped point locations gives the dewarping
transformation.

## tables.json

Contains information about horizontal and vertical dividers that occur
in tabular structures on the dewarped page.

The JSON file contains two keys: `columns` and `dividers`.

`columns` is a dictionary mapping internal names of tabular regions to the
x-positions of all vertical column dividers within that region.

Similarly, `dividers` is a dictionary mapping internal names of tabular regions to
the y-position of all horizontal dividers within that region. 

Example:

```
{
    "version": 1,
    "columns": {
        "regions/TABULAR/5.1.1.1": [2349.5, 2433.6],
        "regions/TABULAR/5.3.1.1": [2349.5, 2433.6]
    },
    "dividers": {
        "regions/TABULAR/14.3.1.1": [705.1, 786.1, 1186.7],
        "regions/TABULAR/6.1.1.1": [1150.7, 1184.9] 
    }
}
```

## order.json

Contains the detected reading order on the dewarped page.

Orders a given as lists of region names. This file can contain various
orders for different configurations, which can be useful.

For example, one might want to have a reading order for `TEXT`
regions, without caring for tabular regions. `order.json` can
store a  specific reading order for this case. This can be useful,
as the  problem of producing a correct reading order that only contains
certain regions is sometimes simpler than producing a reading order that
contains all regions.

The file structure is as follows:

```
interface Order {
    version: number; // currently 1
    orders: { [filter: string] : Array<string>; }
}
```

A typical `order.json` might look something like this.

```
{   "version": 1,
    "orders": {
        "regions/TEXT": ["regions/TEXT/16", "regions/TEXT/15", ...],
        "regions/TABULAR": ["regions/TABULAR/14", "regions/TABULAR/15", ...], 
        "*": ["regions/TEXT/16", "regions/TABULAR/15", ...]
    }
}
```

Here, three configurations `regions/TEXT`, `regions/TABULAR` and
`*` are provided that give different reading orders. `regions/TEXT`
gives a reading order that only contains `TEXT` regions. Likewise,
`regions/TABULAR` gives a reading order for `TABULAR` regions only.

The special configuration called `*` gives a reading order that
contains all regions on the page. This is the default order used
by subsequent stages and should always be present in `order.json` files.

## ocr.zip

Contains the results of line-based OCR processing. For each line geometry,
gives the corresponding text discovered there as `.txt` file. The paths of
these `.txt` files directly correspond to the structure, ordering and indexing
of the line JSON paths in [lines.zip](#lineszip).

Example:

```
regions/
    TEXT/
        0/
            0.txt
            1.txt
        ...
```

The two text files will contain the text discovered in lines `0` and `1` of
region `TEXT/0`, i.e. the first `TEXT` region.

## compose.zip

Compose contains results of the `compose` batch, i.e. the final OCR results.

If a plain text export was performed, this zip file will contain a file called
`page.txt` that contains the plain text OCR of the document, given in the
 detected reading order.

If a Page XML export was performed, this zip file additionally contains a file
called `page.xml`, which is the desired Page XML.
