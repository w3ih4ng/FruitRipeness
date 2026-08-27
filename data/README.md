# Original dataset, kept unchanged

Expected extracted folders: data/raw/Train and data/raw/Test. A single extraction
wrapper such as data/raw/archive/Train and data/raw/archive/Test is also supported.
Keep the images and labels as provided. Do not delete duplicates or relabel images.
The baseline accepts the original Overipe folder spelling in metadata.

The original archive.zip can remain alongside these folders; the baseline ignores
non-image files. The ZIP is only needed if you choose to rerun the optional audit.
Images, archives and generated output are excluded from Git.

Source: https://www.kaggle.com/datasets/asadullahprl/fruits-ripeness-classification-dataset
Source licence shown on Kaggle: CC0. Preserve source attribution in the report.
The uploader's image-level labels are not independent biological ground truth.
