# Logistic Regression from Scratch: Implementation and Comparative Evaluation

This project implements Logistic Regression entirely from scratch using only NumPy, without using scikit-learn's built-in LogisticRegression class for model training.  
The goal is to understand the mathematics behind logistic regression, including sigmoid activation, binary cross-entropy loss, and gradient descent optimization.  
Finally, the performance of the custom implementation is compared against scikit-learn’s LogisticRegression baseline.

---

## 🚀 Project Objectives

- Generate a synthetic binary classification dataset using `make_classification`.
- Implement the Logistic Regression algorithm **from scratch** using only NumPy.
- Code all components manually:
  - Sigmoid function  
  - Linear model  
  - Binary Cross-Entropy cost  
  - Gradient computation  
  - Gradient Descent parameter updates  
- Train the model until convergence or until tolerance is reached.
- Compare performance (accuracy, precision, recall) with sklearn's LogisticRegression.
- Demonstrate understanding of optimization and model evaluation.

---

## 📦 Dataset

**Dataset:** Synthetic binary classification  
**Source:** sklearn.datasets.make_classification  
**Samples:** 500  
**Features:** 5  
**Properties:**
- Linearly separable  
- Noise-free  
- Suitable for logistic regression testing  

The dataset is generated programmatically inside the notebook.

---

## 🧠 Logistic Regression Implementation (Scratch)

Your implementation includes:
- **Forward Propagation**
  \[
  z = w^T x + b,\quad \hat{y} = \sigma(z)
  \]

- **Sigmoid Activation**
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

- **Binary Cross-Entropy Loss**
  \[
  L = -\frac{1}{m} \sum \left[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})\right]
  \]

- **Gradient Calculation**
  \[
  \frac{\partial L}{\partial w} = \frac{1}{m} X^T(\hat{y} - y)
  \]
  \[
  \frac{\partial L}{\partial b} = \frac{1}{m} \sum (\hat{y} - y)
  \]

- **Gradient Descent Update**
  \[
  w := w - \alpha \frac{\partial L}{\partial w},\quad
  b := b - \alpha \frac{\partial L}{\partial b}
  \]

The model stops training based on:
- Maximum number of iterations, OR  
- Loss change below a tolerance (e.g., 1e-6)

---

## 📊 Evaluation Metrics

The following metrics are computed on the test set:

- **Accuracy**
- **Precision**
- **Recall**

These metrics provide a clear understanding of classification performance, especially for balanced datasets.

---

## 🆚 Comparison with Scikit-Learn

A baseline model using:

```python
from sklearn.linear_model import LogisticRegression
```

is trained on the same dataset.

The results of both models are displayed side-by-side in a summary table, allowing easy comparison of performance.

---

## 📁 Project Structure

```plaintext
project/
│── main.ipynb                 # Complete Google Colab notebook
│── README.md
│── report.md
│── requirements.txt
```

---

## 🧑‍💻 How to Run in Google Colab

1. Open the provided `main.ipynb` notebook.  
2. Run each cell in order:
   - Dataset creation  
   - From-scratch Logistic Regression class  
   - Model training  
   - Evaluation  
   - sklearn baseline comparison  
3. The final output will:
   - Display accuracy, precision, recall  
   - Show both models' performance side-by-side

---

## 📝 Deliverables Provided

- Complete, executable Python implementation  
- Gradient Descent-based Logistic Regression from scratch  
- Performance comparison with sklearn  
- Mathematical explanation (see `report.md`)  

---

## 👤 Author

Benasir Fathima
Cultus Internship Project — 2025
