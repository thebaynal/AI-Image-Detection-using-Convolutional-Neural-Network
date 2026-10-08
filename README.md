# AI Image Detection using a Convolutional Neural Network

A TensorFlow/Keras project that classifies an image as **AI generated** or **real** using a convolutional neural network (CNN). It includes a Jupyter notebook for data preparation, training, evaluation, and model export, plus a CustomTkinter desktop interface for single-image predictions.

The notebook contains a saved evaluation with **92.23% accuracy**. This is a historical result from the notebook's test pipeline, not a guarantee of performance on new images or a independently reproduced benchmark.

## Contents

- [Repository structure](#repository-structure)
- [Dataset and preprocessing](#dataset-and-preprocessing)
- [Model and training](#model-and-training)
- [Recorded results](#recorded-results)
- [Installation](#installation)
- [Desktop application](#desktop-application)
- [Training and evaluation](#training-and-evaluation)
- [Prediction example](#prediction-example)
- [Limitations and troubleshooting](#limitations-and-troubleshooting)
- [License](#license)

## Repository structure

```text
.
├── ai_image_detector_training.ipynb  # Training workflow and saved outputs
├── ai_image_detector.py             # Desktop image classification app
├── models/
│   └── aiDetectionV2.h5              # Saved Keras model, managed with Git LFS
└── README.md
```

The following paths are needed locally but are not included in the tracked project files:

```text
data/
├── fake-v2/       # AI-generated training images: class 0
└── real/          # Real training images: class 1
images/
└── default.jpg    # Startup preview required by the desktop app
```

`real` is an example name for the second class directory. Keep exactly two class directories and verify their label order before training.

## Dataset and preprocessing

The notebook references the Kaggle dataset **`superpotato9/dalle-recognition-dataset`** in a commented `kagglehub` download cell. The dataset itself is not bundled with this repository. Obtain it separately, review its terms, and arrange the two classes under `data/`.

The saved notebook output reports **21,587 image files across two classes**, loaded in **675 batches**. The pipeline uses `tf.keras.utils.image_dataset_from_directory('data')`, with observed batches of shape `(32, 256, 256, 3)`, then divides pixel values by 255.

| Setting | Notebook behavior |
| --- | --- |
| Image size | 256 × 256 pixels |
| Channels | 3 (RGB in the directory loader) |
| Batch size | 32 |
| Normalization | Pixel values divided by 255 |
| Labels | 0 = AI generated; 1 = real |
| Training split | `int(675 * 0.7)` = 472 batches |
| Validation split | `int(675 * 0.15) + 1` = 102 batches |
| Test split | `int(675 * 0.15)` = 101 batches |

The split operates on batches using `take()` and `skip()`, rather than a persisted list of image paths. Check `data.class_names` before applying the label mapping; labels are inferred from directory names.

An optional cleanup cell checks formats with OpenCV and `imghdr`. It is commented out and includes deletion of images outside its accepted format list. Review it before enabling it and retain a backup of the dataset.

## Model and training

The notebook contains two architecture variants. Its executable training cell builds this model:

| Layer | Configuration |
| --- | --- |
| Input | `(256, 256, 3)` |
| Convolution | 32 filters, 3 × 3 kernel, ReLU |
| Max pooling | 2 × 2 |
| Dropout | 0.25 |
| Convolution | 64 filters, 3 × 3 kernel, ReLU |
| Max pooling | 2 × 2 |
| Dropout | 0.25 |
| Flatten | Converts feature maps into a vector |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Output | 1 unit, sigmoid |

That cell uses Adam with learning rate `1e-4`, binary cross-entropy, accuracy, balanced class weights calculated with scikit-learn, and a maximum of 50 epochs. Its callbacks are:

- `EarlyStopping`: monitors validation loss, patience 10, restores best weights.
- `ReduceLROnPlateau`: monitors validation loss, factor 0.2, patience 5, minimum learning rate `1e-6`.

A later, commented architecture uses convolution blocks with 16, 32, and 64 filters, batch normalization, pooling, dropout, global average pooling, and a 256-unit dense layer. Its saved summary reports **40,929 parameters**, including **40,705 trainable parameters**. A subsequent executable cell recompiles `model` using default Adam settings.

**Reproducibility note:** the saved summary and evaluation cannot be assumed to describe the executable training cell or the bundled HDF5 model. The notebook retains outputs from an interactive session, and the alternate architecture is currently commented out. Select one architecture and rerun training, summary, and evaluation together to obtain consistent results. The desktop app's caption says “50 epochs” and “92% accuracy”; it is static text rather than model metadata.

## Recorded results

The notebook evaluates batches with Keras `Precision`, `Recall`, and `BinaryAccuracy` and records:

| Metric | Saved value |
| --- | --- |
| Precision | 0.7591463 (75.91%) |
| Recall | 0.8440678 (84.41%) |
| Binary accuracy | 0.9223361 (92.23%) |

With the documented labels, precision and recall refer to the positive class, **real images**. The notebook also contains accuracy/loss plots and individual outside-dataset examples. Those examples include an image named `real.png` predicted as AI generated, so the saved output does not support perfect detection.

## Installation

The notebook records Python **3.12.7**. Python 3.12 is a sensible starting point because both the notebook and application import `imghdr`, which was removed in Python 3.13. No dependency lockfile or tested package version list is provided; the commands below install the libraries imported by the project.

### 1. Clone and retrieve the model

Install Git and Git LFS, then run:

```bash
git lfs install
git clone https://github.com/thebaynal/AI-Image-Detection-using-Convolutional-Neural-Network.git
cd AI-Image-Detection-using-Convolutional-Neural-Network
git lfs pull
```

The model is approximately **378 MB**. Downloading a source ZIP may leave an LFS pointer instead of the actual model. Use Git LFS to retrieve the binary. The repository's committed `.gitattributes` configures `.h5` files for LFS.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux with Python 3.12 installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow numpy opencv-python matplotlib pillow customtkinter scikit-learn jupyterlab
```

Optional, for the notebook's dataset download cell:

```bash
python -m pip install kagglehub
```

Tkinter also needs to be available in your Python installation. The standard Windows Python installer normally includes it; some Linux installations require an additional operating-system package.

## Desktop application

1. Ensure `models/aiDetectionV2.h5` contains the actual downloaded model.
2. Create an `images` directory in the repository root and place a JPEG named `default.jpg` inside it. The application opens this file during startup.
3. Launch from the repository root:

   ```bash
   python ai_image_detector.py
   ```

4. Click **Browse Image** and select a PNG, JPG, or JPEG.
5. Click **Detect**. A Matplotlib window shows the selected image and predicted class.

The interface is a fixed 500 × 400 window with a 256 × 256 preview. Inference reads the image with OpenCV, resizes it with TensorFlow, normalizes it, adds a batch dimension, and predicts a sigmoid score.

| Score | Application label |
| --- | --- |
| Less than 0.5 | AI Generated |
| Greater than or equal to 0.5 | Real Image |

The score is a model output, not a verified confidence percentage. Although a helper variable is named `similarity_percentage`, the application does not calculate a similarity percentage or display it.

## Training and evaluation

1. Download and organize the dataset under `data/` as described above.
2. Start Jupyter from the repository root:

   ```bash
   jupyter lab ai_image_detector_training.ipynb
   ```

3. Select the kernel associated with your virtual environment.
4. Inspect the dataset class order and confirm that AI-generated images map to 0 and real images to 1.
5. Choose the intended model architecture. The executable training cell already calls `model.fit()`; the later alternative architecture and TensorBoard training cells are commented out.
6. Run the loading, scaling, splitting, and chosen training cells, followed by the history plots and test metrics.
7. Replace the notebook's hard-coded external-image paths with files on your machine before running the example inference cells.
8. Create `models/` if needed, then run the export cell:

   ```python
   model.save(os.path.join('models', 'aiDetectionV2.h5'))
   ```

Exporting to this path replaces the local bundled model. Preserve a copy if you want to compare models. Training duration and memory use depend on hardware and the selected architecture; GPU use is optional and is not configured explicitly in the notebook.

## Prediction example

This standalone example reproduces the desktop application's current preprocessing without launching its interface:

```python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('models/aiDetectionV2.h5', compile=False)
image = cv2.imread('example.jpg')
if image is None:
    raise ValueError('Cannot read example.jpg')

image = tf.image.resize(image, (256, 256)) / 255.0
score = float(model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0])
label = 'AI Generated' if score < 0.5 else 'Real Image'
print(f'{label} (raw score: {score:.4f})')
```

OpenCV reads BGR channels, whereas the notebook's directory loader supplies RGB. The current application does not convert BGR to RGB. For a newly trained RGB model, use `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` before resizing and validate predictions with that consistent preprocessing.

## Limitations and troubleshooting

- **Missing startup image:** add `images/default.jpg` before launching the application.
- **Missing or invalid model:** check the working directory and run `git lfs pull`; a small text pointer is not a loadable HDF5 model.
- **Python 3.13+ import failure:** use Python 3.12 with the current source, or update the code to remove/replace `imghdr`.
- **File selection errors:** the current GUI does not handle canceling the file dialog or clicking Detect before choosing an image. Select a valid image first.
- **Color inconsistency:** align BGR/RGB preprocessing between training and inference before relying on results.
- **Unstable dataset partitions:** the directory loader shuffles by default, and the notebook splits the same dataset with independent `take()`/`skip()` pipelines without a fixed seed or persisted membership. Disjoint, repeatable partitions are not guaranteed. Create explicit train/validation/test file lists before using results as a benchmark.
- **Architecture and output mismatch:** saved notebook outputs may be stale. Rerun the chosen model's summary and metrics in a clean session; inspect the loaded model's `summary()` separately.
- **Generalization:** the recorded dataset evaluation does not establish performance across other generators, editing workflows, compression levels, or image sources. Test on independent data representative of your use case.
- **Dependency compatibility:** dependency versions are unpinned. Record working versions when reproducing the project, especially for TensorFlow/Keras and the legacy `.h5` model.

Potential improvements include consistent RGB preprocessing, reproducible splits, a confusion matrix and per-class metrics, independent generator-specific evaluation, GUI input validation, a dependency lockfile, and replacing static performance captions with verified model metadata.

## License

No license file is included in the current repository. Check with the repository owner before redistributing or using the code or model beyond applicable permissions. Dataset terms are separate from code and model permissions.
