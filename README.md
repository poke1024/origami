# origami

Building and using OCR tools can be hard. origami is a low-overhead suite of batch tools for OCR processing to make one’s life easier. It consists of various minimalistic batches and tools useful in building training data for training models and/or generating OCR output from existing models.

# Installation 

```
conda create --name origami python=3.7
conda activate origami
pip install -r origami/requirements.txt

# now install scikit geometry:
conda install scikit-geometry -c conda-forge
```

If you want to run `origami.batch.detect.lines`, you also need to install Tesseract and `tesseract-ocr`.

On Windows, you probably will need to run these:

```
conda install shapely  -c conda-forge
conda install -c anaconda cairo
conda install -c conda-forge tesserocr
```

## General Usage

```
cd /path/to/origami
python -m origami.batch.detect.binarize
```

All command line tools will give you help information on their arguments when called as above.

The given data path should contain processed pages as images. Generated data is put into the same path.  Images may be structured into any hierarchy of sub folders.

Batch processes can be run concurrently.

For generating ground truth for training an OCR engine from a corpus, we suggest this general process:

* Run batches up to `lines` on your page images.
* Sample random lines using `origami.tool.sample`.
* Fine tune your training corpus using `origami.tool.pick`.
* Annotate using `origami.tool.annotate`.
* Annotate using `origami.tool.export`.
* Train your OCR model.

## Batches

<dl>
  <dt>origami.batch.detect.binarize</dt>
  <dd>needs: images</dd>
  <dd>⁂ Binarize all images in the given folder according to the specified arguments using Sauvola binarization.</dd>
</dl>

<dl>
  <dt>origami.batch.detect.segment</dt>
  <dd>needs: images</dd>
  <dd>⁂ Perform segmentation (e.g. separation into text and background) on all images using a neural network model. By default, this uses <a href="https://github.com/poke1024/bbz-segment">origami’s own model.</a>. The predicted classes and labels are embedded in the downloaded model.</dd>
</dl>

<dl>
  <dt>origami.batch.detect.contours</dt>
  <dd>needs: images, binarize, segment</dd>
  <dd>⁂ From the segmentation, detects connected components to produce polygonal contours for blocks and separator lines.  Uses a couple of rule-based approaches to fix some issues inherent in pixel-based segmentation (for details, see command line arguments). The separator detection is too slow and still work in progress.</dd>
</dl>

<dl>
  <dt>origami.batch.detect.lines</dt>
  <dd>needs: images, segment</dd>
  <dd>⁂ Detects baselines and line boundaries for each text line. For details, see  command line arguments. </dd>
</dl>

## Debugging

<dl>
  <dt>origami.batch.debug.contours (debugging only)</dt>
  <dd>needs: images, contours</dd>
  <dd>⁂ Produces debug images for understanding the result of the contours batch stage. <img src="/docs/img/sample-2436020X_1925-02-27_70_98_009.debug.contours.jpg"></dd>
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

