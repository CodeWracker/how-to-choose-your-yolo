# Research on YOLO Model Selection Based on Dataset Characteristics

This repository presents an ongoing research project that aims to **establish a clear methodology for selecting the optimal YOLO model based on dataset characteristics**. Unlike the typical approach of arbitrarily choosing a few YOLO versions (e.g., v3, v5, v7, v9), training them on a large dataset such as COCO, and comparing results, our goal is to measure specific attributes of the dataset—referred to here as “dataset health metrics”—*before* we decide which YOLO version is likely to perform best. This method is designed to save time, computational resources, and increase the scientific rigor of our work.

---

## Motivation

1. **Resource Efficiency**  
   Training large models like YOLO on big datasets (e.g., COCO) is computationally expensive. By diagnosing the dataset beforehand, we can avoid exhaustively training every YOLO variant.

2. **Scientific Rigor**  
   Many research papers compare only two or three YOLO versions without a clear rationale for their choices. Our methodology seeks to provide a **reproducible pipeline** for model selection grounded in measurable dataset characteristics.

3. **Dataset Variability**  
   Real-world datasets can differ significantly: number of classes, distribution of instances per class, spatial distribution of objects, image resolutions, etc. A model that excels on a dataset with uniform class distribution might underperform on a dataset with skewed or clustered distributions. We aim to quantify these effects.

---

## Objectives

1. **Analyze Dataset Characteristics**  
   - Perform a “dataset health” assessment, capturing metrics such as **annotation consistency**, **image dimensions**, **class balance**, **object density**, and **spatial distribution** of objects.

2. **Create Dataset Variants**  
   - Modify the COCO dataset to produce controlled variants:
     - **Number of classes** (e.g., 10, 20, 40, 80)  
     - **Instance distribution** (uniform vs. real-world skewed)  
     - **Instance density per image** (sparse vs. dense)  
     - **Spatial distribution** (concentrated vs. uniformly spread)

3. **Train and Evaluate Multiple YOLO Versions**  
   - Compare YOLOv3, YOLOv5, YOLOv7, YOLOv9 (and possibly others) on each dataset variant.  
   - Use standardized metrics such as **mean Average Precision (mAP)**, **precision**, **recall**, **inference speed (FPS)**, **training time**, and **resource usage**.

4. **Develop a Model Selection Framework**  
   - Correlate dataset metrics with YOLO performance.  
   - Propose guidelines for choosing the YOLO version that is most likely to perform well given the dataset’s measurable characteristics—*before* starting the training process.

---

## Methodology

### 1. Dataset Diagnosis

We begin by **diagnosing the dataset** to ensure we have a numeric understanding of its characteristics. This step yields a set of quantitative metrics that describe:

- **Number of images**  
- **Number of annotations**  
- **Average image size**  
- **Median image ratio**  
- **Number of missing annotations**  
- **Number of null annotations**  
- **Object count histogram**  
- **Heatmap of annotation locations**

Below is an example of how we visualize the results from our **Dataset Health Checker** on the *original* dataset:

| **Class Distribution (Training Dataset)** | **Class Distribution (Validation Dataset)** |
|-------------------------------------------|--------------------------------------------|
| ![Class Distribution Train](dataset_health_checker/results/class_distribution_train.png) | ![Class Distribution Val](dataset_health_checker/results/class_distribution_val.png) |

| **Heatmap of Object Annotations (Training Dataset)** | **Heatmap of Object Annotations (Validation Dataset)** |
|------------------------------------------------------|-------------------------------------------------------|
| ![Heatmap Train](dataset_health_checker/results/heatmap_train.png) | ![Heatmap Val](dataset_health_checker/results/heatmap_val.png) |

> **Why numeric measurements?**  
> Numeric metrics allow us to systematically compare how well different YOLO versions handle variations. If we only describe the dataset qualitatively (e.g., “some classes have lots of instances, some have fewer”), it is hard to replicate or compare across experiments. By converting each characteristic into a **measurable number**, we make the research both **reproducible** and **statistically testable**.

### 2. Creating Dataset Variants

From the original COCO dataset, we create multiple variants:

- **Varying the Number of Classes**  
  - 80 classes (original COCO)  
  - 40 classes  
  - 20 classes  
  - 10 classes  

- **Balancing / Skewing Class Distributions**  
  - For instance, a “uniform” dataset variant where every selected class has the same number of instances.  
  - A “real-world skewed” variant where a few classes dominate.

- **Adjusting Instance Density per Image**  
  - Sparse (few objects per image) vs. dense (many objects).

- **Modifying Spatial Distribution**  
  - Concentrated objects (all bounding boxes near the center) vs. uniform spread (objects appear across different regions of the image).

### 3. Model Training and Evaluation

We train YOLOv3, YOLOv5, YOLOv7, YOLOv9, and possibly additional YOLO variants on each dataset variant. We record:

1. **Mean Average Precision (mAP)**  
2. **Precision**  
3. **Recall**  
4. **Inference Speed (FPS)**  
5. **Training Time**  
6. **GPU/Memory Usage**

These metrics allow us to compare how each YOLO version performs under controlled changes to the dataset, exposing potential strengths and weaknesses in different scenarios.

### 4. Correlation Analysis

After collecting performance metrics for each dataset variant, we explore **statistical relationships** between dataset metrics (e.g., class distribution entropy, spatial distribution metrics) and model performance (e.g., mAP, inference speed). This may involve:

- **ANOVA**  
- **Tukey’s HSD Test**  
- **Regression Models**

### 5. Framework Development

Finally, we use these correlation insights to propose **a reproducible pipeline** that suggests which YOLO version is optimal given a dataset’s numeric profile. For example, the framework might recommend YOLOv9 for datasets with high entropy in class distribution, or YOLOv3 for datasets with fewer classes and sparse objects.

---

## Metrics

Below we list the main “dataset health” metrics we plan to measure. Each is **numeric** and has a clear interpretation, making them suitable for statistical analyses. Our overarching principle is: **if it cannot be expressed numerically, we cannot reliably correlate it with YOLO performance**.

### 1. Class Distribution Metrics

#### 1.1. **Entropy of Class Distribution**

**Reason:**  
Entropy measures the level of **uniformity vs. skewness** in the distribution of objects across classes. A high entropy indicates a more uniform distribution, whereas a low entropy indicates that some classes dominate.

**Equation (Shannon Entropy):**  
$$
H = -\sum_{i=1}^{n} p_i \log(p_i)
$$

Where:
- \(H\) is the entropy.  
- \(p_i\) is the proportion of class \(i\).  
- \(n\) is the total number of classes.  

**Interpretation:**  
- **High \(H\):** Balanced class distribution (no single class is dominant).  
- **Low \(H\):** Distribution is skewed toward a few classes.

#### 1.2. **Gini Index**

**Reason:**  
Often used in economics to measure inequality, the Gini Index can quantify **class imbalance** in a dataset.

**Equation:**  
$$
G = 1 - \sum_{i=1}^{n} (p_i)^2
$$

Where:
- \(G\) is the Gini Index.  
- \(p_i\) is the proportion of class \(i\).  

**Interpretation:**  
- **\(G \approx 0\):** High imbalance (one or few classes dominate).  
- **\(G \approx 1\):** High balance (all classes have similar representation).

**Alternative Approaches:**  
Some researchers use different imbalance metrics like **Coefficient of Variation** or **Imbalance Ratio**. For instance, the Coefficient of Variation of class frequencies can also capture imbalance, but the Gini Index is more standard for capturing distribution inequality.

#### 1.3. **Standard Deviation of Instances per Class**

**Reason:**  
Standard deviation tells us **how dispersed** the class frequencies are.

**Equation:**  
$$
\sigma = \sqrt{\frac{\sum (x_i - \bar{x})^2}{N}}
$$

Where:
- \(\sigma\) is the standard deviation.  
- \(x_i\) is the number of instances in class \(i\).  
- \(\bar{x}\) is the mean number of instances across all classes.  
- \(N\) is the total number of classes.  

**Interpretation:**  
- **Low \(\sigma\):** Class frequencies are nearly the same (balanced).  
- **High \(\sigma\):** Some classes have far more instances than others (skewed).

---

### 2. Spatial Distribution Metrics

While “class distribution” focuses on **how many** instances each class has, “spatial distribution” focuses on **where** objects appear in each image. These metrics help clarify whether objects are **clustered** in certain regions or **spread out** evenly.

#### 2.1. **Entropy of Object Locations**

**Reason:**  
We can measure how uniformly bounding boxes occupy the image. If objects are mostly in one region, the spatial entropy is low; if they are scattered across many regions, the entropy is high.

**Procedure to Calculate:**  
1. Divide each image into a grid (e.g., 10×10).  
2. Count how many bounding boxes (or bounding box centers) fall into each grid cell.  
3. Compute entropy over these proportions:  
   $$
   H = -\sum_{i=1}^{n} p_i \log(p_i)
   $$
   where \(p_i\) is the proportion of objects in grid cell \(i\), and \(n\) is the total number of cells.

**Interpretation:**  
- **High entropy:** Objects spread out across the image.  
- **Low entropy:** Objects concentrated in a few grid cells.

**Alternative Approach:**  
Instead of a fixed 10×10 grid, we could use **adaptive binning** based on image size or average bounding box size. This might give a more stable measure across different resolutions.

#### 2.2. **Standard Deviation of Object Centers**

**Reason:**  
Indicates how widely object centers are dispersed. If \(\sigma_x\) and \(\sigma_y\) are small, objects cluster near a common point; if large, they are spread out.

**Equation (combined metric):**  
$$
D = \sqrt{(\sigma_x)^2 + (\sigma_y)^2}
$$

Where:
- \(\sigma_x\) is the standard deviation of bounding box center \(x\)-coordinates.  
- \(\sigma_y\) is the standard deviation of bounding box center \(y\)-coordinates.  

**Interpretation:**  
- **Low \(D\):** Clustered distribution in image space.  
- **High \(D\):** More uniform or widespread distribution across the image.

#### 2.3. **Distance from Center of Mass**

**Reason:**  
Measures how far objects (on average) are from the image’s center, effectively detecting if there is a “center bias.”

**Equation:**  
$$
D_{cm} = \sqrt{(x_{cm} - x_{img})^2 + (y_{cm} - y_{img})^2}
$$

Where:
- \(x_{cm}, y_{cm}\) is the overall center of mass (mean of all object centers).  
- \(x_{img}, y_{img}\) is the geometric center of the image.

**Interpretation:**  
- **Low \(D_{cm}\):** Most objects are near the image center.  
- **High \(D_{cm}\):** Objects lie farther from the center.

**Alternative Approach:**  
In some cases, we might measure the **distribution** of distances from the center, not just the mean. This could provide additional detail on how objects are spread around the center.

---

## Current Tools

1. **Dataset Conversion Tool**  
   Converts the original COCO dataset to YOLO TXT format to ensure compatibility with various YOLO training pipelines.

2. **Dataset Health Evaluation Tool**  
   - Computes all **class distribution** metrics (e.g., entropy, Gini index, standard deviation).  
   - Examines **annotation completeness** (missing/null annotations).  
   - Assesses **spatial distribution** (object center plots, heatmaps).  
   - Outputs visualizations (histograms, heatmaps) and numeric values for each metric.
