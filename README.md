# Glaucoma Detection — Iris Segmenter

**Status:** Segmentation pipeline (optic-disc segmentation) — U-Net implemented in TensorFlow / Keras.  
**Data:** Training and evaluation datasets are confidential and **not** included in this repository.

---

## Project summary
This repository implements an end-to-end deep-learning pipeline to **segment the optic disc** from cropped optic-disc fundus images. The produced segmentation masks are intended as the foundational step toward automated glaucoma assessment (e.g., cup/disc analysis or a downstream classifier). The core model is a fully convolutional **U-Net** implemented with `tensorflow.keras`; preprocessing and augmentation use OpenCV and NumPy.

---

## What’s included
- `Iris_Segmenter(End_to_End).ipynb` — primary notebook containing the full pipeline: data loading placeholders, preprocessing, augmentation, U-Net model definition, training loop, custom loss, metric computation, evaluation visuals, and inference examples.
- Utilities and cells for visualization of input → mask overlays and metric reporting.

> **Note:** No training or evaluation data is provided here.

---

## Problem statement
Segment the optic disc from cropped RGB optic-disc images to produce a binary per-pixel segmentation mask (background vs disc). This segmentation output is intended for downstream clinically relevant measurements and for feeding into classification models for glaucoma detection.

---

## Approach & pipeline (high level)
- **Input:** Cropped optic-disc RGB images.  
- **Preprocessing:** Resizing and intensity normalization; masks ensured to be binary (0/1).  
- **Augmentation:** Paired image+mask augmentations applied to training samples to improve generalization.  
- **Model:** U-Net encoder–decoder with skip connections (see architecture section).  
- **Loss:** Custom combined loss to handle class imbalance (overlap + pixelwise components).  
- **Metrics:** Pixel-wise accuracy and Dice coefficient (overlap) used for evaluation.  
- **Output:** Predicted probability map per pixel; thresholding yields a binary segmentation mask.

---

## Model architecture (U-Net) — implementation details
This project uses a classic U-Net style architecture implemented in TensorFlow/Keras. Key design choices documented here reflect the implementation in the notebook:

- **Encoder–decoder (fully convolutional) with skip connections** to preserve high-resolution localization information while learning hierarchical features.
- **Initial filter count:** small base (8 filters at first encoder block) and doubles per downsampling stage (e.g., 8 → 16 → 32 → 64), chosen to keep model capacity moderate and memory-efficient for medical images.
- **Convolution blocks:** each encoder/decoder block contains stacked 3×3 convolutions with `ELU` activations and `he_normal` kernel initialization; two convolution layers per block improves representational power.
- **Downsampling:** max-pooling layers are used between encoder blocks to reduce spatial resolution and increase receptive field.
- **Upsampling:** decoder uses learned upsampling (transposed convolutions or upsampling + convolution) followed by concatenation with corresponding encoder feature maps (skip connections) and further convolutional refinement.
- **Output head:** a 1×1 convolution with `sigmoid` activation produces a per-pixel probability for the binary disc class.
- **Regularization options:** dropout layers are available in the notebook to mitigate overfitting for small datasets.
- **Training considerations (not prescriptive):** training in the notebook uses an adaptive optimizer and monitors validation Dice/accuracy to save best checkpoints.

---

## Data preprocessing & augmentation
Robust preprocessing and augmentation are critical for medical segmentation where datasets are often small and variable:

- **Standardization:** Images are resized to a fixed input resolution and normalized (pixel values scaled to [0,1] or standardized per channel).
- **Mask handling:** Masks are binarized and cast to float types to match model outputs.
- **Geometric augmentations (applied identically to image+mask):**
  - Horizontal and vertical flips
  - Small rotations (e.g., ±15°)
  - Random translations and zooms
- **Intensity augmentations:**
  - Brightness/contrast jitter
- **Implementation:** augmentations implemented using OpenCV + NumPy so that spatial transforms preserve mask alignment exactly.

---

## Custom loss & metrics
This project uses a **Jaccard Coefficient Loss** (IoU-based) implemented as a custom loss function and adapted for the U-Net training regime used here. The implementation in the notebook emphasizes overlap optimization while providing flexibility through **adjustable weighting for specific layers**.

**What the Jaccard Coefficient Loss is**
- The Jaccard coefficient (Intersection-over-Union, IoU) measures overlap between predicted and ground-truth masks. The loss used is based on `1 − IoU` (with a small smoothing term to ensure numerical stability).
- IoU-focused loss directly optimizes the overlap metric that matters for segmentation quality, making it a strong choice when the target occupies a small portion of the image (optic disc vs background).

**Why this choice**
- Jaccard loss aligns training with the evaluation objective (overlap), while layer-weighting / deep supervision helps with optimization and faster convergence on encoder–decoder networks.
- The notebook also reports **Dice coefficient** and **pixel-wise accuracy** per epoch for a comprehensive view of model performance: IoU/Dice capture overlap quality, while accuracy gives a coarse pixel-level view (used carefully because background dominates).

---

## Training & evaluation (notes about what's in the notebook)
- The notebook includes training loops and validation evaluation that compute and log accuracy and Dice coefficient per epoch.  
- Model checkpoints (best by validation Dice or validation loss) and visualization cells demonstrate qualitative results: original image, ground truth mask, predicted probability map, and final thresholded mask.  
- Quantitative results are computed locally when the notebook is executed with the private dataset.

---

## Results & intended downstream usage
- **Primary deliverable:** high-quality optic-disc segmentation masks (probability maps + binary masks).  
- **Downstream uses:** segmentation outputs can be used to be fed to a classifier to perform glaucoma vs. non-glaucoma prediction. The segmentation stage reduces variability and provides explicit spatial context for clinical feature extraction.
