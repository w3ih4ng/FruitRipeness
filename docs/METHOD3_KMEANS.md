# Method 3: K-means colour segmentation

## Controlled experiment

This method retains all source images, duplicates and labels. It uses the same
3,547 fit images, 887 validation images, 32 x 32 RGB feature extractor and 300-tree
Random Forest configuration as the baseline, HSV and Otsu runs. Each processing
method trains its own classifier. The original 180-image Test set is not evaluated
by the experiment command. No earlier models need to be retrained.

K-means groups similar pixel colours; its clusters are NOT the three ripeness
classes. This implementation estimates foreground from the clusters and passes the
masked original RGB image to the classifier, not cluster IDs or centroid colours.

## Fixed algorithm

1. Apply EXIF orientation and convert to RGB at the original image resolution.
2. Sample a regular grid of at most 64 rows by 64 columns. Coordinates include both
   endpoints and use floor(linspace(0, dimension-1, min(dimension, 64))). This is
   sampling, not colour interpolation or a change to the source image. Small images
   use every pixel. On 300 x 300 images, fit clustering on 4,096 sampled pixels.
3. Convert the sampled RGB values to float64 in [0, 1]. Fit scikit-learn KMeans with
   k=3, k-means++ initialisation, n_init=3, max_iter=50, tol=0.0001, random_state=42,
   and the Lloyd algorithm. If fewer than three distinct sampled colours exist,
   reduce k accordingly. With only one sampled colour, return an empty mask and
   flag it instead of inventing a foreground/background separation.
4. Sort the resulting centroids by R, then G, then B. Assign EVERY original pixel
   to its nearest centroid using squared Euclidean RGB distance. Equal distances
   choose the first sorted centroid. Assignment runs in bounded chunks of 16,384
   pixels; chunking does not change the calculation.
5. Count cluster membership in the outer image frame. Its width is round(5% of the
   smaller dimension), at least one pixel. Corners are counted once. Treat the
   cluster with the most frame pixels as background. Ties use the largest
   whole-image pixel count, then the first sorted centroid. Retain all other
   clusters as candidate foreground.
6. Apply the same cleanup as HSV/Otsu: one 3 x 3 opening, one 5 x 5 closing, fill
   enclosed holes, and keep every eight-connected component of at least
   max(16 pixels, 0.5% of image area). Use the same edge padding.
7. Keep original RGB values inside the mask; set excluded pixels to black. Do not
   crop, recolour retained pixels or silently fall back to the original image.
8. Use the common RGB feature extractor and Random Forest training/evaluation path.

Clustering is fitted independently to each image, including during inference.
This is an image-only transformation, with no fruit names, filenames, ripeness
labels or other images supplied to it. Algorithm settings were fixed before the
full validation experiment; they were not searched to force an improvement.

Native clustering threads are limited to one during each small fit, then restored.
Random Forest parallelism is unchanged. The threadpoolctl dependency (already used
by scikit-learn) is now explicitly pinned to 3.6.0 because this project imports it.

## Limits to discuss in the report

- RGB distance is affected by lighting and is not perceptually uniform. Clusters
  can separate highlights, shadows or fruit colours rather than fruit/background.
- The border rule assumes one background colour group dominates the frame. Fruit
  touching the edges and multicoloured backgrounds can violate that assumption.
- The fixed sample may miss very small or narrow colour regions. A one-colour
  SAMPLE does not prove the full image is constant. Such inputs are retained with
  an empty mask, and the cluster/sample counts make this case visible.
- k=3 is a fixed exploratory choice, not a proven optimal number of clusters.
- Morphological hole filling can restore background enclosed by a foreground
  region. Fixed pixel kernels also behave differently at different resolutions.
- Empty masks stay in training and evaluation as black images. Mask fraction is
  not segmentation accuracy; the dataset provides no reference masks.
- Validation accuracy may improve or worsen. A change alone does not establish a
  causal explanation; inspect both successful and failed predictions and masks.
- Keep the unchanged dataset's existing limitations in the final report.

## Apply and run

Save method3_kmeans_update.patch beside pyproject.toml in the existing project.
Check first:

```powershell
git apply --check .\method3_kmeans_update.patch
```

If there is no error, run these one at a time, stopping on any error:

```powershell
git apply .\method3_kmeans_update.patch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expect 31 passing tests. Apply the patch only once. Then run:

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.experiment --method kmeans --data data/raw --baseline-run outputs/baseline/20260827T153126_909623Z
```

Use your actual baseline timestamp if different. Before fitting, the command checks
the dataset/split identity against that baseline. Comparison checks common features,
model settings, split and feature-library versions.

## Outputs

The terminal prints a new timestamped folder under outputs/kmeans/. It contains
model.joblib, metadata.json, metrics.json, split_manifest.csv, validation predictions
and confusion matrix, and comparison_validation.csv/.json against the baseline
(overall and per fruit). Existing baseline/HSV/Otsu run folders are not overwritten.

processing_diagnostics.csv includes mask coverage/status and processing time, plus:

| Field | Meaning |
|---|---|
| kmeans_clusters | Effective k after counting distinct sampled colours |
| kmeans_sample_pixels | Number of original pixels sampled for clustering |
| background_cluster | Selected ID after sorting centroids; not a fruit/stage class |
| background_border_fraction | Fraction of frame pixels in the background group, before cleanup |
| kmeans_iterations | Iterations of the selected fit; zero for the one-colour case |

Preview PNGs show original, mask and processed image, actual label/prediction,
foreground coverage, cluster count and background ID. Selection remains the first
validation example for each fruit/stage, plus up to three additional errors.
previews/index.csv records the source and selection reason. Report these selection
rules rather than presenting the previews as a random or exhaustive sample.

Training, preview generation and saved-model inference share process_image. Saved
processing settings must match the implementation. Only load models generated by
your own project; joblib files from untrusted sources are unsafe.

The folder-input comparison UI remains a later milestone; this patch adds Method 3
to the existing experiment runner, not the UI itself.

## Implementation reference

[scikit-learn 1.8 KMeans documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.cluster.KMeans.html)
describes the clustering parameters and their behaviour. Sampling, canonical
centroid ordering, frame-based background selection and morphology are this
project's explicit design choices, not fruit-detection guarantees from K-means.
