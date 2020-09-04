# origami

origami is a self-contained suite of batches and tools for OCR processing of historical newspapers.
It covers many essential steps in a digitization pipeline, including (1) building training data for
training models, and (2) generating Page-XML OCR output from pages using trained models.

Some of origami's features are:

* easy setup, easy to use
* DNN segmentation
* dewarping
* reading order detection
* simple table support
* Page-XML export

Additional tools for:

* annotating ground truth
* debugging
* creating annotated images

# Installation 

```
conda create --name origami python=3.7 -c defaults -c conda-forge --file origami/requirements/conda.txt
conda activate origami
pip install -r origami/requirements/pip.txt
```

## General Usage

```
cd /path/to/origami
python -m origami.batch.detect.segment
```

All command line tools will give you help information on their arguments when called as above.

The given data path should contain processed pages as images. Generated data is put into the same path.  Images may be structured into any hierarchy of sub folders.

## Batches

Given an OCR model, the necessary order of batches for performing OCR for a folder of documents is:

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

Each batch creates and depends upon various files, as shown in the following table:

<table class="table table-header-rotated">
<thead>
<tr>
<th></th>
<th class="rotate"><div><span>page.jpg</span></div></th>
<th class="rotate"><div><span>segment.zip</span></div></th>
<th class="rotate"><div><span>segment.jpg</span></div></th>
<th class="rotate"><div><span>contours.0.zip</span></div></th>
<th class="rotate"><div><span>flow.zip</span></div></th>
<th class="rotate"><div><span>lines.0.zip</span></div></th>
<th class="rotate"><div><span>contours.1.zip</span></div></th>
<th class="rotate"><div><span>dewarp.zip</span></div></th>
<th class="rotate"><div><span>contours.2.zip</span></div></th>
<th class="rotate"><div><span>tables.json</span></div></th>
<th class="rotate"><div><span>contours.3.zip</span></div></th>
<th class="rotate"><div><span>lines.3.zip</span></div></th>
<th class="rotate"><div><span>order.json</span></div></th>
<th class="rotate"><div><span>ocr.zip</span></div></th>
<th class="rotate"><div><span>compose.zip</span></div></th>
</tr>
</thead>
<tbody>
<tr>
<th class="row-header">segment</th>
<td>&#9711;</td>
<td>&#11044;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<th class="row-header">contours</th>
<td></td>
<td></td>
<td>&#9711;</td>
<td>&#11044;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<th class="row-header">flow</th>
<td>&#9711;</td>
<td></td>
<td></td>
<td>&#9711;</td>
<td>&#11044;</td>
<td>&#11044;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<th class="row-header">dewarp</th>
<td></td>
<td></td>
<td></td>
<td>&#9711;</td>
<td>&#9711;</td>
<td></td>
<td>&#11044;</td>
<td>&#11044;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<th class="row-header">layout</th>
<td>&#9711;</td>
<td>&#9711;</td>
<td></td>
<td>&#9711;</td>
<td></td>
<td>&#9711;</td>
<td>&#9711;</td>
<td></td>
<td>&#11044;</td>
<td>&#11044;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<th class="row-header">lines</th>
<td>&#9711;</td>
<td>&#9711;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>&#9711;</td>
<td>&#9711;</td>
<td>&#11044;</td>
<td>&#11044;</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<th class="row-header">order</th>
<td></td>
<td>&#9711;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>&#9711;</td>
<td></td>
<td>&#9711;</td>
<td></td>
<td>&#9711;</td>
<td>&#9711;</td>
<td>&#11044;</td>
<td></td>
<td></td>
</tr>
<tr>
<th class="row-header">ocr</th>
<td>&#9711;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>&#9711;</td>
<td></td>
<td>&#9711;</td>
<td></td>
<td>&#11044;</td>
<td></td>
</tr>
<tr>
<th class="row-header">compose</th>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>&#9711;</td>
<td>&#9711;</td>
<td>&#9711;</td>
<td>&#9711;</td>
<td>&#9711;</td>
<td>&#11044;</td>
</tr>
</tbody>
</table>

## Default Batches

<dl>
  <dt>origami.batch.detect.segment</dt>
  <dd>needs: images</dd>
  <dd>⁂ Perform segmentation (e.g. separation into text and background) on all images using a neural network model. By default, this uses <a href="https://github.com/poke1024/bbz-segment">origami’s own model.</a>. The predicted classes and labels are embedded in the downloaded model.</dd>
</dl>

<dl>
  <dt>origami.batch.detect.contours</dt>
  <dd>needs: images, binarize, segment</dd>
  <dd>⁂ From the segmentation, detects connected components to produce vectorized polygonal contours for blocks and separator lines. 
  Uses a couple of rule-based approaches to fix some issues inherent in pixel-based segmentation (see --region-spread,
  --ink-spread and --ink-opening for details). Note that the separator detection is very slow and still work in progress.</dd>
</dl>

<dl>
  <dt>origami.batch.detect.lines</dt>
  <dd>needs: images, segment</dd>
  <dd>⁂ Detects baselines and line boundaries for each text line. For details, see  command line arguments. </dd>
</dl>

<dl>
  <dt>origami.batch.detect.ocr</dt>
  <dd>needs: images, lines</dd>
  <dd>⁂ Performs OCR on each detected line using the specified Calamari OCR model. Note that the binarization
  you can specify here in independent of the one performed in origami.batch.detect.binarize.</dd>
</dl>

<dl>
  <dt>origami.batch.detect.order</dt>
  <dd>needs: images, contours</dd>
  <dd>⁂ Tries to find a reading order using a variant of the XY Cut algorithm.</dd>
</dl>

<dl>
  <dt>origami.batch.detect.compose</dt>
  <dd>needs: images, lines, ocr, xycut</dd>
  <dd>⁂ Composes text into one file using the detected reading order.</dd>
</dl>

<dl>
  <dt>origami.batch.detect.stats</dt>
  <dd>needs: nothing</dd>
  <dd>⁂ Prints out statistics on computed artifacts and errors. This is useful for
  understanding how many pages for processed, and for which stages this processing
  is finished.</dd>
</dl>

## Debugging

<dl>
  <dt>origami.batch.annotate.contours</dt>
  <dd>needs: stages 1, 2 (and maybe more, depending on `--stage`)</dd>
  <dd>⁂ Produces debug images for understanding the result of the contours batch stage.
  <img src="/docs/img/sample-2436020X_1925-02-27_70_98_009.debug.contours.jpg"></dd>
</dl>

<dl>
  <dt>origami.batch.annotate.lines</dt>
  <dd>needs: stages 1 - 6</dd>
  <dd>⁂ Produces debug images for understanding the line detection stage.
  <img src="/docs/img/sample-SNP2436020X-18720410-1-12-0-0.lines.jpg">
  </dd>
</dl>

<dl>
  <dt>origami.batch.annotate.layout</dt>
  <dd>needs: stages 1 - 7</dd>
  <dd>⁂ Produces debug images for understanding the result of the layout and order
  batch stage.</dd>
</dl>

## Tools

<dl>
  <dt>origami.tool.annotate</dt>
  <dd>needs: images, lines</dd>
  <dd>⁂ Tool for annotating, viewing and searching for ground truth. <img src="/docs/img/sample-annotation.jpg"></dd>
</dl>

<dl>
  <dt>origami.tool.pick</dt>
  <dd>needs: images, lines</dd>
  <dd>⁂ Tool for adding or removing single lines from the ground truth for fine tuning. <img src="/docs/img/sample-linepick.jpg"></dd>
</dl>

<dl>
  <dt>origami.tool.sample</dt>
  <dd>needs: images, lines</dd>
  <dd>⁂ Create a new annotation database by randomly sampling lines from a corpus. The details of sampling (numbers of items
  for each segmentation label type per page) can be specified. Allows import of transcriptions stored in accompanying PageXML.
  See command line help for more details.</dd>
</dl>

<dl>
  <dt>origami.tool.schema</dt>
  <dd>⁂ Run an annotation normalization schema on the given ground truth text files.</dd>
</dl>

<dl>
  <dt>origami.tool.export</dt>
  <dd>⁂ From the given annotation database, export line images of the specified height and binarization together with accompanying
    ground truth text files. Annotation normalization through a schema is supported. Use this command to generate training data for
    <a href="https://github.com/Calamari-OCR/calamari">Calamari</a>. See command line for details.</dd>
</dl>

<dl>
  <dt>origami.tool.xycut</dt>
  <dd>⁂ Debug internal X-Y cut implementation.</dd>
</dl>

<dl>
  <dt>origami.batch.export.lines (debugging only)</dt>
  <dd>needs: images, lines</dd>
  <dd>⁂ Export images of lines detected during lines batch.</dd>
</dl>

<dl>
  <dt>origami.batch.export.pagexml  (debugging only)</dt>
  <dd>needs: images, lines</dd>
  <dd>⁂ Export polygons of lines detected during lines batch as PageXML.</dd>
</dl>


## How to create ground truth

For generating ground truth for training an OCR engine from a corpus, we suggest this general process:

* Run batches up to `lines` on your page images.
* Sample random lines using `origami.tool.sample`.
* Fine tune your training corpus using `origami.tool.pick` (optional).
* Annotate using `origami.tool.annotate`.
* Export annotations using `origami.tool.export`.
* Train your OCR model.

## Concurrency

Batch processes can be run concurrently. Origami supports file-based locking or by using a database (see `--lock-strategy`). The latter strategy is more compatible and set by default.
Use `--lock-database` to specify the path to a lock database (if none is specified, Origami will create one in your data folder).

## Dinglehopper

To evaluate performance using Dinglehopper, you probably want to use:

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
