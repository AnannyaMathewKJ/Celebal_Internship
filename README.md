# Data Science Internship
#  Week 1 Task
A thorough foundational set of tasks encompassing fundamental data engineering and mathematical concepts needed for deep machine learning implementations may be found in this repository. In order to concentrate on the underlying structural code mechanics, each section stays away from opaque black-box framework logic.

## 📁 Repository Structure
Math assertions, structural visualizations, and full code implementations are all contained in the core Google Colab workspace pipeline.

## 🛠️ Detailed Implementations
### 1. Fundamentals of Python and Defensive Programming
* **Control Flow Automation:** Clean, boundary-validated conditional pathways were used to formulate numerical evaluation algorithms.
* **Algorithmic Optimization:** Advanced list comprehensions and sets were used for optimized memory operations, while dynamic string frequencies were processed manually to avoid standard library structures.
* **Exception Layer Design:** Robust, runtime-safe division handlers were created to raise explicit feedback alerts (`TypeError`) when non-numeric parameters are caught and to securely intercept arithmetic anomalies (`ZeroDivisionError`).

### 2. Using NumPy for Scientific Computing
* **Multi-Dimensional Transformations:** From initial 1D vectors to highly targeted 3D multi-layered matrices (`2x2x3`), dimensional mutations were performed across structural representations.
* **Matrix Operations & Invariance Checking:** Element-wise array operations and conventional inner dot-products were assessed, and non-commutative spatial behavior ($P \times Q \neq Q \times P$) was explicitly demonstrated.

### 3. Pandas Data Wrangling and Feature Profiling
* **Relational Query Slicing:** Absolute indexing techniques (`.loc` vs. `.iloc`) based on composite masking operations were used to isolate important target parameters.
* **Production Preprocessing Pipelines:** Developed full missing value recovery tracks using dynamic localized statistics (mean-shift standard approximations for age metrics, and median value column imputations for target outliers).

### 4. Dimensionality Reduction and Linear Algebra
* **Euclidean Norm Calculations:** Used spatial quiver visualizations to compute matrix $L_2$ characteristics and visualize continuous directional coordinates.
* **Spectral Analysis (Eigenpair Extraction):** linear configurations that have been broken down to produce basic transformation attributes (Eigenvalues & Eigenvectors). created assertion layers that confirm the basic identity vector relation:
$$A\vec{v} = \lambda\vec{v}$$ Singular Value Decomposition was used in Data Matrix Compression (SVD) to separate arbitrary observation profiles into primary coordinates ($U, \Sigma, V^T$). To mimic structural PCA patterns, low-rank approximations were performed utilizing leading dominating single elements.

### 5. Exploration and Descriptive Statistics
* **Distribution Characterization:** Complete numerical profiles (Mean, Median, Standard Deviation, and Interquartile Ranges) were computed for continuous fields in order to characterize the distribution.
* Visualizations of Density: To map production variances versus typical normal profiles, kernel density estimates (KDE) were plotted over frequency histograms.

  
---

## 💻 Technical Setup & Stack
* **Language Environment:** Python 3 (Google Colab Environment)
* **Core Packages:** `numpy`, `pandas`, `scipy.stats`, `matplotlib`, `seaborn`

# Week 2 Task

## 📌 Project Overview

This project designs and implements an end-to-end Machine Learning and Statistical Time Series pipeline to predict **Tesla's Monthly Global Vehicle Deliveries**. Utilizing real-world operational tracking datasets spanning from 2015 through 2025, the workflow evaluates predictive capabilities under strict out-of-sample chronological forecasting boundaries.

The primary engineering focus of this project was the structural refactoring of an initial high-performing regression model to eliminate a severe case of **Target Leakage / Look-Ahead Bias**, ensuring true predictive utility in real-world deployment scenarios.

---

## 🛠️ Tech Stack & Dependencies

* **Data Analysis & Profiling:** `Pandas`, `NumPy`
* **Data Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning Models:** `Scikit-Learn` (Gradient Boosting Regressor, Random Forest Regressor)
* **Statistical Modeling:** `Statsmodels` (Holt-Winters Exponential Smoothing)
* **Data Ingestion:** `KaggleHub`

---

## 🚀 Pipeline Architecture

### 1. Ingestion & Preprocessing

* Automated raw file downloads directly through `kagglehub`.
* Cleaned whitespace inconsistencies in structural column headers.
* Converted fractional time metrics (`Year`, `Month`) into a standardized `DatetimeIndex`.
* Aggregated granular, regional, and model-specific entries to compute continuous **Unified Global Monthly Totals**.

### 2. Exploratory Data Analysis (EDA)

* Profiled selling price distributions using Kernel Density Estimate (KDE) histograms.
* Analyzed total production volume distribution trends categorized by specific vehicle classes.
* Generated a numeric metric correlation matrix to examine multi-collinear interactions between production output and consumer deliveries.

### 3. Feature Engineering & Target Transformation

To construct a valid forecasting environment, the feature matrix was updated using two major techniques:

* **Operational Feature Lagging:** Key predictors (`Production_Units`, `Avg_Price_USD`) were lagged by 1 month ($\text{Lag}_1$), ensuring future target points are calculated exclusively using verified historical context.
* **Stationary Target Differencing:** To overcome the mathematical limitation where tree-based algorithms struggle to extrapolate outside their historical absolute training scale limits, the target metric was transformed into a **Month-over-Month Change** series:

$$\Delta Y_t = Y_t - Y_{t-1}$$



### 4. Validation Framework

* **Chronological Splitting:** Avoided standard randomized shuffling (`train_test_split`) which breaks sequential dependencies.
* **Holdout Test Set:** Segmented the final 12 unique months (**2025-01 to 2025-12**) as an unseen holdout window.
* **Cross-Validation:** Applied `TimeSeriesSplit` cross-validation during the optimization loops to secure parameters without look-ahead contamination.

---

## 🚨 Resolving the Target Leakage Trap

During initial baseline evaluations, standard regression setups yielded an artificial **$R^2$ Score of 0.86 to 0.95**. Technical audit revealed that this high performance was a structural flaw caused by target leakage:

1. Concurrent production units for the same target month were available in the feature matrix.
2. Because Tesla manufactures vehicles to fulfill active order queues, current production has a **0.99 correlation** with current deliveries.

### The Solution:

By explicitly dropping all current-month indicators from the input matrix, shifting operational metrics back by 1 month, and training a **Tuned Gradient Boosting Regressor** exclusively on differenced variations, the look-ahead bias was entirely eliminated. The predicted changes were reconstructed back into true absolute figures during post-processing using the preceding month's actual baseline:


$$\hat{Y}_{\text{Absolute}} = Y_{t-1} + \hat{\Delta Y}_t$$

---

## 📊 Final Performance Scoreboard (2025 Validation Horizon)

| Model Strategy | RMSE | MAE | $R^2$ Score |
| --- | --- | --- | --- |
| **Tuned Gradient Boosting (Stationary Pipeline)** | 14,865.62 | 13,039.42 | **-0.34** |
| **Holt-Winters Baseline (Time Series)** | 11,741.93 | 9,483.22 | **0.17** |

### Key Strategic Insights:

* **Statistical Resilience:** The classical Holt-Winters model achieved a positive $R^2$ score of **0.17**, outperforming machine learning models by cutting through month-over-month noise to model Tesla's long-term 12-month cyclical corporate delivery loops.
* **Volatility Limits:** Pure history-based autoregressive lags cannot anticipate sudden real-world global market adjustments (such as the sharp V-shaped plunge observed in May 2025) without external leading indicators.

---

## 🔮 Future Enhancements (Phase 2 Roadmap)

1. **Exogenous Feature Integration:** Incorporate global macroeconomic signals, including consumer price indices (CPI), electric vehicle battery raw material costs, and regional interest rate updates.
2. **Hybrid Model Architectures:** Implement deep learning sequence models (`LSTM`, `GRU`) paired with linear trend-residual structural decompositions.

# **Week 3 Task**
## 📌 Project Overview

The objective of this project is to categorize entities (countries/macro-profiles) into distinct socio-economic or behavioral segments and build a predictive system capable of reverse-engineering those segments using classification models. 

The system operates in two core phases:
1. **Unsupervised Profiling:** Identifies natural clusters within the multi-dimensional dataset based on financial, health, and development metrics.
2. **Supervised Actionable Intelligence:** Uses tree-based ensemble models to predict these cluster assignments and maps feature importance to discover the primary drivers behind the segmentation boundaries.

---

## 📊 Dataset Profile

* **Source:** Kaggle Unsupervised Learning on Country Data
* **Dimensions:** 167 rows, 10 columns
* **Data Integrity:** 0 missing values, 0 duplicate rows detected
* **Key Features:** `child_mort`, `exports`, `health`, `imports`, `income`, `inflation`, `life_expec`, `total_fer`, `gdpp`

---

## ⚙️ Pipeline Architecture

### 1. Data Ingestion & Preprocessing
* Automated dataset download integrated directly via `kagglehub`.
* Non-numeric identifiers (`country`) are isolated, and numeric features are scaled using `StandardScaler` to handle distance distortion in clustering.

### 2. Clustering Analysis
* **K-Means Clustering:** Optimized using the Elbow Method and Silhouette Analysis over a range of $k \in [2, 10]$[cite: 1]. An optimal split of $k=3$ groups data into strategic tiers.
* **DBSCAN Clustering:** Used alongside K-Means to analyze localized continuous density space and flag extreme behavioral outliers/noise[cite: 1].

### 3. Supervised Classification & Interpretation
* The data is partitioned into a 75/25 train-test split, preserving cluster proportions via stratification.
* **Random Forest Classifier:** Trained with 100 estimators to establish a solid baseline map of cluster geometry.
* **XGBoost Classifier:** Deployed to maximize predictive convergence and extract gradient-boosted feature importance maps.

---

## 📈 Model Performance & Insights

### Performance Summary
* **Random Forest:** Achieved an overall classification accuracy of **98%** on unseen test data.
* **XGBoost:** Reached an accuracy profile of **98%**, perfectly capturing the geometric boundaries carved out by the clustering algorithms.

### Actionable Driving Factors
According to the tree-based split logs, the top predictors driving the definition of profile segments are[cite: 1]:
1. **`child_mort`** (Importance Score: ~0.5376)
2. **`life_expec`** (Importance Score: ~0.1042)
3. **`gdpp`** (Importance Score: ~0.0958)
4. **`income`** (Importance Score: ~0.0938)

---

## 🛠️ Installation & Usage

To run this project directly in a Google Colab or local Jupyter notebook environment, ensure you have the following dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub
```
# Week 4 Task

## 📌 Project Overview
This project focuses on the implementation and comparative evaluation of deep learning architectures for image classification using the **CIFAR-10 dataset**. The workflow transitions from a baseline Artificial Neural Network (ANN) to a deep Convolutional Neural Network (CNN) to demonstrate how preserving spatial structures and utilizing targeted regularization optimization pathways dramatically improves model generalization and performance.

---

## 🛠️ Tech Stack & Dependencies
* **Deep Learning Framework:** `TensorFlow`, `Keras`
* **Data Processing & Manipulation:** `NumPy`, `Pandas`
* **Visualization & Performance Analytics:** `Matplotlib`, `Seaborn`

---

## 🚀 Pipeline Architecture

### 1. Ingestion & Preprocessing
* Automated loading of the CIFAR-10 dataset ($32 \times 32 \times 3$ color images across 10 target classes).
* Normalized raw pixel intensity channels to a $[0, 1]$ scale to stabilize gradient steps during backpropagation.
* Implemented operational **Data Augmentation** techniques (including random rotations, horizontal flips, and width/height shifts) to artificially expand training sample diversity and reduce overfitting.

### 2. Baseline Model: Artificial Neural Network (ANN)
* **Structural Design:** Composed of fully connected (`Dense`) hidden layers.
* **Limitation:** Requires flattening the 3D image arrays into a single 1D vector of 3,072 features ($32 \times 32 \times 3$). This destroys the local pixel topology and spatial relationships within the image, causing a massive parameter explosion that leads to rapid overfitting.

### 3. Advanced Model: Convolutional Neural Network (CNN)
* **Structural Design:** Built using alternating Convolutional (`Conv2D`) layers and Max Pooling (`MaxPooling2D`) dimensions.
* **Mechanism:** Employs sliding kernels to capture local receptive fields, systematically learning a hierarchical representation of features—ranging from low-level edges and textures to complex high-level shapes and object patterns.
* **Efficiency:** Leverages weight sharing across spatial coordinates to structurally minimize the total parameter count compared to the fully connected alternative.

### 4. Regularization & Optimization Layer
* **Batch Normalization:** Scaled hidden layer activations to mitigate internal covariate shift, leading to faster convergence and smoother loss landscapes.
* **Dropout:** Randomly deactivated a set percentage of neurons during training batch iterations to break up co-dependency patterns and force the network to learn robust, redundant features.
* **Early Stopping:** Configured structural validation monitoring loops to track performance and safely halt training updates before validation error began to diverge.

---

## 📊 Comparative Performance & Insights

| Evaluation Vector | Artificial Neural Network (ANN) | Convolutional Neural Network (CNN) |
| :--- | :--- | :--- |
| **Spatial Awareness** | None (Flattens structural inputs into 1D) | High (Preserves 2D/3D spatial neighborhood topology) |
| **Parameter Efficiency** | Inefficient (Susceptible to parameter explosion) | Highly Efficient (Utilizes translation-invariant weight sharing) |
| **Generalization Gap** | High (Prone to memorizing training set noise) | Low (Maintains balanced validation boundaries) |
| **Classification Accuracy**| Baseline performance constraints | Significantly superior overall accuracy and stability |

### Key Technical Takeaways:
1. **Topology Preservation:** Traditional ANNs are inherently limited for computer vision problems because flattening operations discard essential spatial context.
2. **Optimization Engineering:** Deep architectures require an intentional application of **Data Augmentation**, **Batch Normalization**, and **Dropout** to manage variance and perform effectively on unseen, real-world data distributions.
3. **Core Competence:** This project completes the foundational pipeline requirements for deep learning deployments, demonstrating end-to-end fluency in hyperparameter tuning, debugging training curves, and structural neural network optimization.
# 🚀 Data Science Internship - Celebal Technologies

This repository documents the weekly tasks, implementations, and core learning objectives completed during my Data Science Internship track.

---

## 📅 Week 5: Fine-Tuning and Transfer Learning

### 🧠 Core Concepts
* **Transfer Learning:** Reusing feature maps from state-of-the-art pre-trained architectures (e.g., ResNet, VGG, MobileNet) optimized on massive datasets like ImageNet to drastically reduce training time.
* **Fine-Tuning:** Unfreezing specific top layers of a pre-trained network and training them jointly with the new custom classification layers to adapt to domain-specific features.
* **Data Augmentation:** Introducing random transformations (rotation, scaling, shearing, horizontal flips) to the training pipeline to combat overfitting on small datasets.

### 🛠️ Implementation Checklist
- [x] Loaded a pre-trained backbone model without its top classification layer.
- [x] Froze the base convolutional layers to preserve the learned feature extractions.
- [x] Appended custom Fully Connected (Dense) layers matching the target dataset classes.
- [x] Compiled the model using an appropriate optimizer with a conservative learning rate.
- [x] Trained and evaluated the newly added classification head.

---

## 📅 Week 6: Image Denoising using Autoencoders

### 🎯 Objective
Build and train a **Convolutional Autoencoder** deep learning model using Keras/TensorFlow to successfully remove synthetic noise (Gaussian/Salt-and-Pepper) from the **MNIST digits dataset**.

### 🏗️ Architectural Concept
An autoencoder is an unsupervised neural network that compresses an input into a lower-dimensional latent space representation (**Encoder**) and then reconstructs the original input from this representation (**Decoder**). 

For a **Denoising Autoencoder (DAE)**, the input is intentionally corrupted with noise, while the target loss is calculated against the original, clean image. This forces the network to learn the underlying structural manifold of the digits rather than simply memorizing identity mappings.

```text
       ┌─────────────────┐       ┌───────────────────┐       ┌─────────────────┐
       │   Noisy Input   │ ────> │   Encoder Layers  │ ────> │  Latent Space   │
       │  (Digit + Noise)│       │  (Conv2D + MaxPool│       │ (Bottleneck Z)  │
       └─────────────────┘       └───────────────────┘       └─────────────────┘
                                                                      │
       ┌─────────────────┐       ┌───────────────────┐                │
       │ Denoised Output │ <──── │   Decoder Layers  │ <──────────────┘
       │ (Clean Reconst.)│       │ (Conv2D + UpSample)│
       └─────────────────┘       └───────────────────┘
