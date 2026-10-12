# Mathematical Facial Re-Identification

YOLO face detection → grayscale 64 × 64 crop → normalized 4096-vector →
PCA eigenfaces → Mahalanobis and Riemannian matching → fused score → identity or Unknown.

Run commands from this project folder. The existing `venv` contains the dependencies.
For a fresh environment, create one with `python -m venv venv`, then install with
`.\venv\Scripts\python.exe -m pip install -r requirements.txt`.
The face checkpoint is `models/yolov8n-face.pt`; its provenance is in `models/README.md`.

## 1. Enroll people

```powershell
.\venv\Scripts\python.exe main.py enroll --name Ayush --samples 10
.\venv\Scripts\python.exe main.py enroll --name Person2 --samples 10
```

Click the enrollment window. Press **S** to save one face and **Q** to finish.
Exactly one face must be visible when saving. Use several distinct frames with
slightly different expressions and angles, keeping lighting and face position consistent.
The session ends after the requested number of new samples. Names are folder-safe labels;
`Unknown` is reserved. New sessions append crops instead of overwriting them.

For image input:

```powershell
.\venv\Scripts\python.exe main.py enroll --name Ayush --images photo1.jpg photo2.jpg photo3.jpg
```

Training requires at least two identities and three samples per identity; ten or more
distinct samples per identity are preferable for this prototype. Crops are stored as
`data/train/<identity>/<unique-id>.png`. Source frames are not stored.

## 2. Train the gallery

```powershell
.\venv\Scripts\python.exe main.py train --components 20 --alpha 0.5
```

This saves `models/gallery.npz`. PCA automatically limits the requested component count
to the numerical rank of the centered training samples. Re-run training whenever enrollment
changes; the previous calibration is replaced by a provisional threshold.

## 3. Recognize

```powershell
.\venv\Scripts\python.exe main.py identify
.\venv\Scripts\python.exe main.py identify --image query.jpg
```

Live recognition draws an identity and fused distance for each face; press **Q** to close.
The image command prints JSON results; an empty list means no face was detected.
Lower scores mean a closer match, not a probability. `--threshold 1.0` overrides the
stored acceptance threshold for an experiment. `--camera 1` selects another camera;
`--device cpu` forces CPU inference. `main.py camera` runs just the camera preview;
`python src/detector.py` still runs the detection/preprocessing preview.

## 4. Calibrate Unknown with separate faces

Capture new frames of enrolled people and of at least one person who is **not** in training:

```powershell
.\venv\Scripts\python.exe main.py enroll --name Ayush --split validation --samples 5
.\venv\Scripts\python.exe main.py enroll --name Person2 --split validation --samples 5
.\venv\Scripts\python.exe main.py enroll --name Visitor --split validation --samples 5
.\venv\Scripts\python.exe main.py calibrate --max-false-accept-rate 0.05
```

The initial threshold comes from training leave-one-out scores using a PCA and metric
fitted on all training faces. It is provisional and optimistic. Calibration uses only
validation scores without retraining PCA, the metric, or distance normalization scales.
It targets the 95th percentile of correctly matched known validation scores, lowering
the threshold if necessary to satisfy the requested empirical unknown false-accept rate.
Calibration fails if there are no correctly matched known faces, no unknown faces, or
no feasible threshold. Exact copies of training crops are rejected in calibration/evaluation.
The target is a validation-set operating point, not a guarantee for new people.

## 5. Evaluate on a third, untouched split

Capture separate `--split test` samples of enrolled identities and new unknown people,
then run:

```powershell
.\venv\Scripts\python.exe main.py evaluate
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Evaluation reports the known identification rate (correct identity and accepted),
unknown rejection rate, and false-accept rate. Use validation for parameter selection
and test for the final report; do not reuse captures across splits. Tests use synthetic
images to verify numerical behavior; they do not establish real face-recognition accuracy.

## Mathematics and implementation

- **PCA:** center the row-wise sample matrix `X`; thin SVD gives right singular vectors
  as covariance eigenvectors. Covariance eigenvalues are `s²/(n−1)`. Projection is
  `z = U.T @ (x−mean)`. This avoids allocating the full 4096 × 4096 covariance matrix.
- **Mahalanobis:** estimate a pooled within-identity covariance in PCA space, shrink it
  toward a scaled identity matrix, and compute `sqrt(delta.T @ C^-1 @ delta)` with
  a Cholesky solve. Regularization handles the small-sample, singular-covariance case.
- **SPD descriptor:** each face provides pixel observations `[x, y, I, dI/dx, dI/dy]`.
  Their 5 × 5 covariance plus a positive ridge is SPD, even for flat crops. This is a
  spatial descriptor per image, not a covariance estimated from a single PCA vector.
- **Riemannian distance:** affine-invariant distance is the Euclidean norm of the logs
  of the generalized eigenvalues solving `B v = lambda A v`.
- **Fusion:** divide each distance by its fixed positive median within-identity training
  pair distance, then compute `D = alpha*dM/scaleM + (1−alpha)*dR/scaleR` for each
  enrollment sample. An identity's score is its best sample score. Choose the lowest
  identity score and accept only if it does not exceed the threshold. Scales are fixed
  at training time, so a poor query cannot become a good match merely through query-wise
  min-max normalization. Scale estimation uses up to 30 training samples per identity.

These are raw pixel eigenfaces and spatial covariance descriptors. Lighting, pose,
alignment, and loss of information during PCA can affect identity decisions. The current
preprocessing has no landmark alignment or liveness detection. Validate performance with
your own held-out captures before making recognition claims.

Numerical references: [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
and [SciPy generalized symmetric eigensolver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html).
Enrollment images and the learned gallery are ignored by Git. Gallery persistence uses
numeric arrays and JSON metadata with pickle disabled.
