# Uploaded dataset audit

**Current project decision:** retain the dataset unchanged, including duplicates and
original labels/split. This historical audit documents limitations; its cleaning
recommendations are not the adopted workflow and do not block baseline training.

Audited file: `archive.zip`, supplied in this conversation. Date: 27 August 2026.
Every image was decoded with Pillow. All paths and declared labels were checked.
Exact duplicate detection used SHA-256 of both original bytes and EXIF-normalised
RGB pixels (including dimensions). Perceptual candidate detection used a 64-bit DCT
hash with Hamming distance <= 4. Candidate pairs are NOT automatically deleted.

## Counts before cleaning

| Fruit | Unripe | Ripe | Overripe | Total |
|---|---:|---:|---:|---:|
| Apple | 369 | 331 | 144 | 844 |
| Banana | 472 | 609 | 361 | 1,442 |
| Mango | 607 | 525 | 148 | 1,280 |
| Orange | 136 | 120 | 99 | 355 |
| Tomato | 308 | 301 | 84 | 693 |
| **Total** | **1,892** | **1,886** | **836** | **4,614** |

Original Train: 4,434 images. Original Test: 180 images (60 per maturity stage).
All 4,614 images decoded successfully and are 300 x 300 pixels. A successful decode
does not establish correct semantic labels or independent samples.

## Problems found

1. The training folder uses `Overipe`, while the test folder uses `Overripe`.
   The audit normalises metadata without modifying the original files.
2. 157 byte-identical duplicate groups; 158 decoded-pixel-identical groups.
   Pixel groups contain 216 redundant copies: 4,398 unique decoded images remain
   if retaining one representative per exact pixel group. This is NOT the final
   usable-image count; near duplicates and semantic problems remain.
3. 17 exact-pixel groups cross the supplied train/test boundary. The original split
   therefore contains demonstrated leakage and should not be used unchanged.
4. 296 perceptually similar candidate pairs excluding exact pixel duplicates:
   27 cross-split pairs and 17 pairs with different fruit/stage labels. These are
   overlapping categories, not additive counts or counts of unique images.
5. No exact-pixel duplicate group has conflicting labels, but visual review of
   selected perceptual candidates confirms differently labelled versions of the
   same apparent source photo. Examples below require quarantine/review, not
   automatic selection of one label.
6. A contact sheet of 90 samples (six per fruit/stage) shows varied backgrounds,
   repeated collection settings, multiple fruits per image, cut fruit and visible
   spoilage in some images labelled overripe. This is a sample review, not a full
   independent re-annotation of all 4,614 labels.
7. The archive contains only images and directory/filename labels. It supplies no
   bounding boxes, segmentation masks, blemish masks, physical scale references
   or verified per-fruit specimen identifiers.

## Reproducible examples

### Exact duplicate crossing train/test

- `Test/Overripe/apple_overripe_005.png`
- `Train/Overipe/apple_overripe_045.png`
- `Train/Overipe/apple_overripe_051.png`

These three files share identical normalised RGB pixels.

### Visually confirmed near-duplicate label conflicts

- `Test/Ripe/tomato_ripe_239.jpeg`
- `Train/Overipe/tomato_overripe_035.jpeg`
- `Train/Ripe/tomato_ripe_270.jpeg`

The images appear to be the same tomato scene, with versions labelled ripe and
overripe. The first also crosses train/test relative to the others.

- `Train/Ripe/mango_ripe_296.jpg`
- `Train/Unripe/mango_unripe_039.jpg`

These appear to show the same mango photograph, labelled both ripe and unripe.
Visual similarity alone cannot decide which maturity label is biologically correct.

## Verdict

The dataset is a usable candidate for a bounded educational comparison, but is NOT
ready for trustworthy training/evaluation as supplied. First review conflicts,
deduplicate/group related scenes, decide how to handle multi-fruit images and
rebuild a common split. Report class imbalance and any reduced coverage honestly.
Do not present the original 4,614 files as 4,614 independent observations.

This audit does not prove that all duplicate scenes were detected. Source/physical
fruit IDs are unavailable, so residual leakage risk must be disclosed. Nor can
RGB images establish internal maturity, sweetness, edibility or food safety.

## Re-run

From the project root after installation:

```powershell
.\.venv\Scripts\python.exe -m fruitripeness.audit --zip data/raw/archive.zip
```

The complete manifest and duplicate candidate lists are generated under
`outputs/audit/`. Those generated files and dataset images are not committed.

Source: https://www.kaggle.com/datasets/asadullahprl/fruits-ripeness-classification-dataset
