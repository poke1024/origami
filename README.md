# Origami

Origami is a self-contained suite of batches and tools for OCR processing of historical newspapers.
It covers many essential steps in a digitization pipeline, including (1) building training data for
training models, and (2) generating Page-XML OCR output from pages using trained models.

Apart from its specific features, Origami is

* easy to setup
* easy to use
* based on file-based intermediary results that allow customization

Origami's current default implementation features:

* DNN segmentation
* dewarping
* reading order detection
* simple table support
* Page-XML export

Origami also provides additional tools for:

* annotating ground truth
* debugging
* creating annotated images
* evaluation of OCR quality

## Installing Origami
We provide two options for Installing Origami:

* Run in a Docker container.
* Install and run directly on your machine (in a conda environment).

## Installing with Docker

1. Download and install Docker.
2. Install the NVIDIA container toolkit (necessary for GPU usage).  See [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for installation instructions.
3. Build the docker container (**NOTE**: this process can take ~20 minutes or more, as the container
   builds Scikit-Geometry from source.):

        cd docker
        docker buildx build -t "origami:origami-gpu" .
   This creates a docker image `origami:origami-gpu`.

4. Launch the container.  You must specify the location of your local copy of the Origami repo, as
   shown below:

        docker run --gpus all -it --rm -v /the/local/path/to/origami/:/origami origami:origami-gpu bash
   This runs the container and presents you with an interactive shell, ready to run Origami (located
   in `/origami/`).
   
    _NOTE: Origami requires some additional set-up to run (e.g., downloading the segmentation
   models).  [See below](#the-detection-batches) for details._


## Installing Locally

### Basics

```
conda create --name origami python=3.7 -c defaults -c conda-forge --file origami/requirements/conda.txt
conda activate origami
pip install -r origami/requirements/pip.txt
```

## Troubleshooting scikit-geometry 

On some systems (e.g. macOS 10.15.7) the `conda` installation of scikit-geometry is broken. In these cases,
you can always build scikit-geometry from scratch, i.e.:

```
conda activate origami
git clone https://github.com/scikit-geometry/scikit-geometry
cd scikit-geometry
python setup.py install
```

## General Usage

```
cd /path/to/origami
python -m origami.batch.detect.segment
```

All command line tools will give you help information on their arguments when called as above.

The given data path should contain processed pages as images. Generated data is put into the same path.  Images may be structured into any hierarchy of sub folders.

# Batches

## Artifacts

Origami's processing happens in separated stages, with batches that read and write
information from well-defined files (also called artifacts). Each batch creates
and depends upon various artifacts, as shown in the following
table. Rows depict artifacts, columns depict detection batches (i.e. the batches
found under `origami.batch.detect`). Blank circles indicate a read, filled
circles indicate a write. As illustrated here, later batches depend on information
provided by earlier batches.

Click on the names of the artifacts (left column) or batches (top row) below to get
more information.

| |[segment](#segment)|[contours](#contours)|[flow](#flow)|[dewarp](#dewarp)|[layout](#layout)|[lines](#lines)|[order](#order)|[ocr](#ocr)|[compose](#compose)|
|-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|[page image](docs/formats.md#page-image)|&#9711;| |&#9711;| |&#9711;|&#9711;| |&#9711;| |
|[segment.zip](docs/formats.md#segmentzip)|&#11044;|&#9711;| | |&#9711;|&#9711;|&#9711;| | |
|[contours.0.zip](docs/formats.md#contours0zip)| |&#11044;|&#9711;|&#9711;|&#9711;| | | | |
|[flow.zip](docs/formats.md#flowzip)| | |&#11044;|&#9711;| | | | | |
|[lines.0.zip](docs/formats.md#lines0zip)| | |&#11044;| |&#9711;| | | | |
|[contours.1.zip](docs/formats.md#contours1zip)| | | |&#11044;|&#9711;| |&#9711;| | |
|[dewarp.zip](docs/formats.md#dewarpzip)| | | |&#11044;|&#9711;| | | | |
|[contours.2.zip](docs/formats.md#contours2zip)| | | | |&#11044;|&#9711;|&#9711;| | |
|[tables.json](docs/formats.md#tablesjson)| | | | |&#11044;|&#9711;| |&#9711;|&#9711;|
|[contours.3.zip](docs/formats.md#contours3zip)| | | | | |&#11044;|&#9711;| |&#9711;|
|[lines.3.zip](docs/formats.md#lines3zip)| | | | | |&#11044;|&#9711;|&#9711;|&#9711;|
|[order.json](docs/formats.md#orderjson)| | | | | | |&#11044;| |&#9711;|
|[ocr.zip](docs/formats.md#ocrzip)| | | | | | | |&#11044;|&#9711;|
|[compose.zip](docs/formats.md#composezip)| | | | | | | | |&#11044;|

## Running Batches

### Order

Given an OCR model, and as illustrated in the table from last section, the
necessary order of detection batches for performing OCR for a folder of
documents is:

<table>
<tr>
<td>1</td>
<td>segment</td>
</tr>
<tr>
<td>2</td>
<td>contours</td>
</tr>
<tr>
<td>3</td>
<td>flow</td>
</tr>
<tr>
<td>4</td>
<td>dewarp</td>
</tr>
<tr>
<td>5</td>
<td>layout</td>
</tr>
<tr>
<td>6</td>
<td>lines</td>
</tr>
<tr>
<td>7</td>
<td>order</td>
</tr>
<tr>
<td>8</td>
<td>ocr</td>
</tr>
<tr>
<td>9</td>
<td>compose</td>
</tr>
</table>

### Concurrency

Batch processes can be run concurrently. Origami supports file-based locking or by using a database (see `--lock-strategy`). The latter strategy is more compatible and set by default.
Use `--lock-database` to specify the path to a lock database (if none is specified, Origami will create one in your data folder).

### Modifying Results

It is possible to replace Origami pipeline stages/batches by custom implementations
by simply reading and writing Origami's artifacts using the documented file formats.

It is also possible to run Origami stages and then postprocess the generated artifacts
before continuing with later stages.

## The Detection Batches

### segment

<dl>
  <dt>origami.batch.detect.segment</dt>
  <dd>Performs segmentation (e.g. separation into text and background) on all images
  using a neural network model.
  <br>
  If you have not trained a custom model, you should download and use
  <a href="https://github.com/poke1024/bbz-segment">origami’s default model</a>.
  You need to specify the path to that downloaded model via the `--model` argument
  when calling `origami.batch.detect.segment`.
  <br>
  The predicted classes and labels are embedded in the specified model.</dd>
</dl>

### contours

<dl>
  <dt>origami.batch.detect.contours</dt>
  <dd>From the pixelwise segmentation information, detects connected components to produce vectorized polygonal contours for blocks and separator lines.</dd>
</dl>

### flow

<dl>
  <dt>origami.batch.detect.flow</dt>
  <dd>Detects baselines and warping in separators to produce an overall description of page curvature.</dd>
</dl>

### dewarp

<dl>
  <dt>origami.batch.detect.dewarp</dt>
  <dd>Creates a dewarping transformation that is used in subsequent stages.</dd>
</dl>

### layout

<dl>
  <dt>origami.batch.detect.layout</dt>
  <dd>Refines regions by fixing over- and under-segmentation via heuristic rules. </dd>
</dl>

### lines

<dl>
  <dt>origami.batch.detect.lines</dt>
  <dd>Detects baselines and line boundaries for each text line.</dd>
</dl>

### order

<dl>
  <dt>origami.batch.detect.order</dt>
  <dd>Finds a reading order using a variant of the XY Cut algorithm.</dd>
</dl>

### ocr

<dl>
  <dt>origami.batch.detect.ocr</dt>
  <dd>Performs OCR on each detected line using the specified Calamari OCR model.
  For more details on OCR models, see the
  <a href="https://github.com/poke1024/origami#origami-models">section on Origami OCR models.</a>.</dd>
</dl>

### compose

<dl>
  <dt>origami.batch.detect.compose</dt>
  <dd>Composes text into one file using the detected reading order. Can also produce PageXML output.</dd>
</dl>

## Debugging

<dl>
  <dt>origami.batch.detect.stats</dt>
  <dd>Prints out statistics on computed artifacts and errors. This is useful for
  understanding how many pages for processed, and for which stages this processing
  is finished.</dd>
</dl>

<dl>
  <dt>origami.batch.annotate.contours</dt>
  <dd>Produces debug images for understanding the result of the contours batch stage.
  <img src="/docs/img/sample-2436020X_1925-02-27_70_98_009.debug.contours.jpg"></dd>
</dl>

<dl>
  <dt>origami.batch.annotate.lines</dt>
  <dd>Produces debug images for understanding the line detection stage.
  <img src="/docs/img/sample-SNP2436020X-18720410-1-12-0-0.lines.jpg">
  </dd>
</dl>

<dl>
  <dt>origami.batch.annotate.layout</dt>
  <dd>Produces debug images for understanding the result of the layout and order
  batch stage.</dd>
</dl>

# Tools for Ground Truth and Evaluation

## Tools

<dl>
  <dt>origami.tool.annotate</dt>
  <dd>Tool for annotating, viewing and searching for ground truth. <img src="/docs/img/sample-annotation.jpg"></dd>
</dl>

<dl>
  <dt>origami.tool.pick</dt>
  <dd>Tool for adding or removing single lines from the ground truth for fine tuning. <img src="/docs/img/sample-linepick.jpg"></dd>
</dl>

<dl>
  <dt>origami.tool.sample</dt>
  <dd>Create a new annotation database by randomly sampling lines from a corpus. The details of sampling (numbers of items
  for each segmentation label type per page) can be specified. Allows import of transcriptions stored in accompanying PageXML.
  See command line help for more details.</dd>
</dl>

<dl>
  <dt>origami.tool.schema</dt>
  <dd>⁂ Run an annotation normalization schema on the given ground truth text files.</dd>
</dl>

<dl>
  <dt>origami.tool.export</dt>
  <dd>From the given annotation database, export line images of the specified height and binarization together with accompanying
    ground truth text files. Annotation normalization through a schema is supported. Use this command to generate training data for
    <a href="https://github.com/Calamari-OCR/calamari">Calamari</a>. See command line for details.</dd>
</dl>

<dl>
  <dt>origami.tool.xycut</dt>
  <dd>Debug internal X-Y cut implementation.</dd>
</dl>

<dl>
  <dt>origami.batch.export.lines (debugging only)</dt>
  <dd>Export images of lines detected during lines batch.</dd>
</dl>

<dl>
  <dt>origami.batch.export.pagexml  (debugging only)</dt>
  <dd>Export polygons of lines detected during lines batch as PageXML.</dd>
</dl>

## How to create ground truth

For generating ground truth for training an OCR engine from a corpus, we suggest this general process:

* Run batches up to `lines` on your page images.
* Sample random lines using `origami.tool.sample`.
* Fine tune your training corpus using `origami.tool.pick` (optional).
* Annotate using `origami.tool.annotate`.
* Export annotations using `origami.tool.export`.
* Train your OCR model.

## Origami Models

For line-based OCR, Origami uses Calamari internally and therefore can be used with any Calamari model.

However, Origami's way of segmenting lines is slightly different from other pipelines: lines are not binarized and they are not scaled horizontally (therefore they might be wider than what some models are trained on).

One model specifically trained for Origami is the model used to perform OCR on the
Berliner Börsen-Zeitung. The model (and more context on its training) is available
under https://github.com/poke1024/origami_models

Another suitable model is the <a href="https://qurator-data.de/calamari-models/GT4HistOCR/2019-12-11T11_10+0100/model.tar.xz"> GT4HistOCR model for Calamari</a>. Note
that you need to enable binarization in the OCR for the latter.

## Evalulation via Dinglehopper

To evaluate performance using <a href="https://github.com/qurator-spk/dinglehopper">Dinglehopper</a>, you probably want to use:

```
python -m origami.batch.utils.evaluate DATA_PATH
```

Alternatively, you can create PAGE XMLs manually:

```
python -m origami.batch.detect.compose DATA_PATH \
    --page-xml --only-page-xml-regions \
    --regions regions/TEXT \
    --ignore-letters "{}[]"
```
