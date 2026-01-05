# Fine-Grained Image Classification using FGVC-Aircraft-2013b

This project focuses on **fine-grained image classification** using deep learning, where the goal is to distinguish between visually similar classes with subtle differences. The implementation is part of an academic assignment and explores preprocessing, model training, and evaluation on a challenging real-world dataset.

## Dataset

**FGVC-Aircraft-2013b**

- Domain: Aircraft model classification  
- Number of classes: 100 aircraft variants  
- Images: ~10,000 high-resolution images  
- Task: Fine-grained classification (high intra-class similarity, low inter-class variation)

⚠️ **Dataset not included in this repository**

The dataset is approximately **3 GB** in size and cannot be uploaded to GitHub.  
Please download it directly from the official source:

👉 https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

After downloading, extract and place the dataset in the appropriate local directory as described below.

## Project Structure

```

.
├── COMP8430_Assignment_1.ipynb   # Main Jupyter Notebook
├── data/                        # FGVC-Aircraft dataset (local only, not tracked)
│   ├── images/
│   ├── variants.txt
│   ├── images_variant_train.txt
│   ├── images_variant_test.txt
│   └── images_variant_val.txt
└── README.md

````

## Key Concepts Covered

- Fine-grained visual classification challenges
- Dataset exploration and preprocessing
- Train / validation / test splits
- Deep learning model training
- Model evaluation and performance analysis

## Model & Techniques

- Deep learning–based image classification
- Feature extraction and normalization
- Performance evaluation on unseen test data
- Error analysis for fine-grained categories

*(Specific model details and experiments are documented inside the notebook.)*

## How to Run

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-name>
````

2. Download and extract **FGVC-Aircraft-2013b** from the official website.

3. Place the dataset in a local `data/` directory (path may be adjusted inside the notebook).

4. Open the notebook:

```bash
jupyter notebook COMP8430_Assignment_1.ipynb
```

5. Run cells sequentially.

## Requirements

* Python 3.x
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Deep learning framework (as used in notebook)

## Notes

* This project is intended for **educational and research purposes**.
* Dataset usage follows the terms specified by the original authors.
* The notebook contains all explanations, experiments, and results.

## Acknowledgements

* FGVC-Aircraft Dataset — University of Oxford, Visual Geometry Group
* Dataset creators and maintainers for providing a high-quality benchmark for fine-grained classification

```
