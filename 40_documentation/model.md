# [model.py](../30_experiments/model.py)

Defines two model architectures for oil slick detection: BaselineCNN (lightweight CNN from scratch) and TerraMindClassifier (pre-trained GFM backbone with custom head).

## Overview

A comprehensive model module containing two distinct architectures for binary oil slick classification on Sentinel-1 SAR imagery. Implements a lightweight CNN trained from scratch and a transfer learning approach using a pre-trained TerraMind geospatial foundation model backbone with a custom classification head.

## Core Components

### Weight Initialization

* **`init_kaiming_weights(m)`** Custom weight initialization function:
  - **Conv2d layers**: Kaiming normal initialization (fan_out, relu)
  - **Linear layers (multi-output)**: Kaiming normal initialization (relu)
  - **Linear layers (single output)**: Xavier normal initialization
  - **Biases**: Constant initialization to 0
  Ensures stable gradient flow and faster convergence

## Model Architectures

### BaselineCNN

A lightweight convolutional neural network trained from scratch on SAR imagery.

**Architecture:**

| Layer | Type | Config | Output Shape |
|-------|------|--------|--------------|
| Input | - | 2 channels | [B, 2, 224, 224] |
| Conv2d | 2→16 channels | kernel=3, pad=1 | [B, 16, 224, 224] |
| ReLU | Activation | - | [B, 16, 224, 224] |
| MaxPool2d | Pooling | kernel=2 | [B, 16, 112, 112] |
| Conv2d | 16→32 channels | kernel=3, pad=1 | [B, 32, 112, 112] |
| ReLU | Activation | - | [B, 32, 112, 112] |
| MaxPool2d | Pooling | kernel=2 | [B, 32, 56, 56] |
| AdaptiveAvgPool2d | Global pooling | output=1×1 | [B, 32, 1, 1] |
| Flatten | Reshape | - | [B, 32] |
| Linear | FC layer | 32→64 | [B, 64] |
| ReLU | Activation | - | [B, 64] |
| Dropout | Regularization | p=0.3 | [B, 64] |
| Linear | Output layer | 64→1 | [B, 1] |

**Key Features:**
- Lightweight design for computational efficiency
- Only 2 convolutional blocks (16, 32 channels)
- Global average pooling for translation invariance
- Dropout for regularization
- Single output logit (binary classification via sigmoid)

**Parameters:**
- `num_classes`: Number of output classes (default: 2)
- `in_channels`: Number of input channels (default: 2 for VV, VH)

**Initialization:** Kaiming normal with relu nonlinearity

### TerraMindClassifier

Transfer learning approach using a pre-trained geospatial foundation model backbone.

**Architecture:**

| Component | Details | Notes |
|-----------|---------|-------|
| **Backbone** | TerraMind v1 Small | Pre-trained on Sentinel-1 GRD |
| **Modality** | S1GRD | Sentinel-1 Ground Range Detected |
| **Frozen** | Yes (default) | Backbone parameters not updated |
| **Output Embedding** | Backbone final layer | Varies with model size |
| **Global Pooling** | Mean pooling (if 3D) | Reduces temporal/spatial dimensions |
| **Classification Head** | Custom 2-layer MLP | Trainable fine-tuning head |

**Classification Head:**

| Layer | Type | Config | Output Shape |
|-------|------|--------|--------------|
| Input | Backbone embeddings | - | [B, embedding_dim] |
| Linear | FC layer | embedding_dim→64 | [B, 64] |
| ReLU | Activation | - | [B, 64] |
| Dropout | Regularization | p=0.3 | [B, 64] |
| Linear | Output layer | 64→1 | [B, 1] |

**Key Features:**
- Leverages pre-trained geospatial knowledge
- Frozen backbone prevents catastrophic forgetting
- Custom trainable head for task-specific fine-tuning
- Flexible handling of backbone output shapes
- Mean pooling for 3D feature tensors

**Parameters:**
- `num_classes`: Number of output classes (default: 2)
- `freeze_backbone`: Whether to freeze backbone weights (default: True)

**Initialization:** Kaiming normal for head layers only

## Model Comparison

| Aspect | BaselineCNN | TerraMindClassifier |
|--------|-------------|-------------------|
| **Training Strategy** | From scratch | Transfer learning |
| **Backbone** | Custom CNN | Pre-trained TerraMind |
| **Backbone Frozen** | N/A | Yes |
| **Parameters** | ~50K | ~50K (head only) |
| **Training Epochs** | 32 | 16 |
| **Learning Rate Range** | 1e-4 to 1e-3 | 5e-4 to 3e-3 |
| **Weight Decay Range** | 1e-4 to 1e-2 | 1e-5 to 1e-3 |
| **Expected Advantage** | Lightweight, fast | Leverages geospatial prior |

## Forward Pass Details

### BaselineCNN Forward Pass
1. Input: [B, 2, 224, 224]
2. Feature extraction via convolutional blocks
3. Global average pooling to [B, 32]
4. Classification head (FC layers with ReLU, dropout)
5. Output: [B, 1] (logit for binary classification)

### TerraMindClassifier Forward Pass
1. Input: [B, 2, 224, 224]
2. Backbone feature extraction → embeddings
3. Handle output shape (dict/list/tuple conversion)
4. Reduce dimensionality (mean pooling if 3D)
5. Classification head (FC layers with ReLU, dropout)
6. Output: [B, 1] (logit for binary classification)

## Loss Function

Both models use **BCEWithLogitsLoss** (Binary Cross-Entropy with Logits):
- Combines sigmoid activation with BCE loss (numerically stable)
- Supports `pos_weight` for class imbalance handling
- Single output logit per sample

## Dependencies

* **Internal:** `terratorch.registry` (for TerraMindClassifier only)
* **External:** `torch`, `torch.nn`
