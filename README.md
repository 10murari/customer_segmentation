# 🛍️ Customer Segmentation Using Clustering (K-Means & Hierarchical)

This project applies **K-Means** and **Hierarchical Clustering** to segment customers based on their demographics and spending patterns. It includes an interactive **Streamlit web app** that allows users to visualize clusters and explore customer segments.

---

## 📊 Dataset

We use the **Mall Customer Segmentation Data** containing the following fields:

- `CustomerID`
- `Gender` (mapped: Male = 1, Female = 2)
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

Dataset Source: [Kaggle - Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)

---

## 🧠 Techniques Used

- **Data Preprocessing** with `pandas` and `StandardScaler`
- **K-Means Clustering**
  - Elbow method to determine optimal clusters
- **Hierarchical Clustering**
  - Dendrogram to decide number of clusters
- **Visualization** using `matplotlib`, `seaborn`, and `Streamlit`

---

## 🌐 Streamlit Web App

### Features:
- View full dataset (scrollable)
- Automatically handle missing values
- Apply both K-Means and Hierarchical clustering
- View Elbow plot and Dendrogram
- Visualize clustering results with pairplots

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/10murari/customer_segmentation.git
cd customer_segmentation
