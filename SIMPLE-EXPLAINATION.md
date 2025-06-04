# 🧬 RNA 3D Structure Prediction

## 📌 Overview

This project predicts the **3D structure of RNA molecules**—specifically the x, y, and z coordinates of each nucleotide—from their sequence and base-pairing patterns. Our goal is to help scientists understand RNA folding behavior using **machine learning** models trained on real experimental data.

## 🔍 What Does This Do?

Given an RNA molecule's:
- Nucleotide **sequence** (A, U, G, C),
- **Base pairing probabilities** (how likely two bases are to pair),
- **Secondary structure** (who’s paired with whom),
- And optional **text descriptions** (type, conditions, etc.),

this model predicts:
- **3D coordinates** of nucleotides,
- For some tasks, it may also infer **residue identity**, **residue position**, and **confidence** in structure.

It works on both **short sequences** and **longer RNAs**.

## 🧠 How It Works (Simple Terms)

We use a machine learning model trained to “learn” from examples of RNA sequences with known 3D shapes.

- It “reads” the sequence like a language.
- It “looks” at which nucleotides are paired together (structure).
- It understands patterns from descriptions (e.g., ribozyme, aptamer).
- It combines all this to **predict how the RNA folds in space**.

Think of it as giving the model a script (sequence) and stage directions (base pairs), and it predicts how the RNA would physically act (3D shape).

## 🧪 What Data is Used?

We use a modified version of the **Stanford RNA 3D Folding Dataset**, which contains:
- Thousands of RNA sequences,
- Their secondary structure annotations,
- Base pairing probabilities (from chemical probing or computation),
- Ground truth 3D coordinates (from crystallography or modeling).

The data is split into:
- Training: Used to teach the model.
- Validation: Used to tune and test accuracy.
- Test: RNA sequences with **hidden structures** to see how well the model generalizes.

## 📁 Files and Folders

| File/Folder | Purpose |
|-------------|---------|
| `train_sequences.modified.csv` | RNA sequences + structure info for training |
| `train_labels.modified.csv`    | 3D positions (x, y, z) for each nucleotide |
| `test_sequences.csv`           | Sequences where structure must be predicted |
| `models/`                      | Contains the model logic |
| `train.py`                     | Script to train the model |
| `predict.py`                   | Script to run predictions on test RNA |
| `README.md`                    | You are here! |

## ▶️ How to Use It (Simplified)

You don’t need to know coding to understand what happens. But here's a summary of the workflow:

1. **Load the RNA sequence** and structure from the dataset.
2. The model **analyzes the sequence, base pairs, and metadata**.
3. It predicts the **3D shape** (x, y, z) of each nucleotide.
4. You can **visualize the predicted structure** using 3D tools like PyMOL or Chimera.

To run the prediction:
```bash
python predict.py --input test_sequences.csv --output predicted_structures.csv
````

## 📉 How Good Is It?

The model has been trained and tested on thousands of RNA molecules and shows strong agreement with known experimental structures. It performs especially well on:

* Short to medium-length RNAs,
* Canonical base-pairing patterns,
* Sequences with rich secondary structure information.

We evaluate it using:

* **Mean Error in Coordinates** (how far predictions are from truth),
* **Pearson correlation** (how well trends match true structure),
* **Structural consistency** (does it fold in a physically possible way?).

## 🔬 Why This Matters

RNA structure determines function. With limited availability of high-resolution experimental 3D structures, this tool helps:

* **Speed up RNA drug discovery** by modeling candidate structures,
* **Design synthetic RNAs** with desired shapes,
* **Understand mutations** affecting folding and function.

## 👩‍🔬 Who Should Use This?

This tool is made for:

* **Biologists** working in structural genomics or RNA research,
* **Bioinformaticians** needing 3D RNA input for downstream analysis,
* **Synthetic biologists** designing riboswitches or RNA circuits,
* **Students** learning about RNA folding mechanisms.

No deep ML knowledge needed to apply this—just RNA sequences and an interest in understanding how they behave in 3D!

## 🧰 Requirements (if you want to run it locally)

* Python 3.8+
* PyTorch (used for deep learning)
* PyTorch Geometric (for graph-based modeling of RNA structure)
* Optional: DeepSpeed (for faster training)

Kaggle notebooks are already set up with everything ready.

---

## 📞 Questions or Collaboration?

Feel free to reach out or raise an issue if:

* You want to adapt this for specific RNAs,
* You have better structure datasets,
* You’re curious how machine learning can help in biology!

---

## 💡 Final Note

This model is **not a crystal ball**, but it can offer **meaningful structural hypotheses** for RNAs with no solved 3D structure. It's a step toward making RNA biology more accessible, faster, and scalable using AI.

Happy folding! 🧬
