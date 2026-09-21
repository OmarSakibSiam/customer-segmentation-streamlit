# Customer Segmentation Using K-Means

A Streamlit web application that assigns a customer to a segment using a trained **K-Means clustering model**.

The machine-learning workflow was developed in `Customer_segmentation_project.ipynb`. The notebook:

- cleans the customer dataset,
- creates customer-related features,
- selects seven features for clustering,
- standardizes those features with `StandardScaler`,
- trains a K-Means model with **6 clusters**,
- saves the trained scaler and model with `joblib`.

The Streamlit application loads the saved `.pkl` files and predicts the cluster for new customer information.

---

## Features Used by the Model

The feature order must remain exactly the same as during training:

| Order | Feature | Meaning |
|---:|---|---|
| 1 | `Age` | Customer age |
| 2 | `Income` | Customer income |
| 3 | `Total_spend` | Total spending across the product categories used in the notebook |
| 4 | `NumWebPurchases` | Number of purchases made through the website |
| 5 | `NumStorePurchases` | Number of purchases made in stores |
| 6 | `Recency` | Days since the customer's most recent purchase |
| 7 | `NumWebVisitsMonth` | Number of website visits per month |

In the notebook:

```python
features = [
    "Age",
    "Income",
    "Total_spend",
    "NumWebPurchases",
    "NumStorePurchases",
    "Recency",
    "NumWebVisitsMonth"
]
```

---

## Project Structure

Your GitHub repository should look like this:

```text
customer-segmentation/
│
├── app.py
├── requirements.txt
├── README.md
├── scaler.pkl
├── k_means_model.pkl
└── Customer_segmentation_project.ipynb   # optional, but recommended
```

### Model filename note

The notebook saves the K-Means model as:

```text
k_means_model.pkl
```

The supplied `app.py` also supports:

```text
k_mean_model.pkl
```

So the application works even if you renamed the model file.

---

## How the App Works

The Streamlit application follows the same preprocessing sequence used during training:

```text
Customer input
      ↓
Create 7-feature DataFrame
      ↓
scaler.pkl
      ↓
Standardized values
      ↓
k_means_model.pkl
      ↓
Predicted cluster
```

The app also displays a simple profile of the predicted cluster by comparing the cluster centroid with the standardized training-data average.

> K-Means is an unsupervised algorithm. A cluster number such as `0`, `1`, or `5` is only an identifier; it is not a score, rank, probability, or confidence value.

---

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

Streamlit will show a local URL in the terminal, usually:

```text
http://localhost:8501
```

---

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload these files to the repository:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `scaler.pkl`
   - `k_means_model.pkl` or `k_mean_model.pkl`
3. Push/commit the files to GitHub.
4. Sign in to **Streamlit Community Cloud**.
5. Choose **Create app** / **Deploy an app**.
6. Select your GitHub repository and branch.
7. Set the main file path to:

```text
app.py
```

8. Deploy the application.

No API key or Streamlit secrets are required for this project.

---

## Important: scikit-learn Version Compatibility

`joblib`/pickle model files can depend on the version of `scikit-learn` that created them.

The Streamlit deployment currently pins the versions known from the training environment:

```text
pandas==2.2.3
numpy==2.1.3
joblib==1.6.0
```

The Python import is named `sklearn`, but the installable package is named
`scikit-learn`. If a package-version checker reports `sklearn: Not Found`,
that does not necessarily mean scikit-learn is missing.

Check the exact training version with:

```python
import sklearn
print(sklearn.__version__)
```

or:

```python
from importlib.metadata import version
print(version("scikit-learn"))
```

Until that exact version is known, `requirements.txt` uses:

```text
scikit-learn>=1.5,<2.0
```

For maximum pickle compatibility, replace that line with the exact training
version once you know it:

```text
scikit-learn==YOUR_EXACT_VERSION
```

If Streamlit Cloud reports an `InconsistentVersionWarning` or an error while loading the model, matching the training version of scikit-learn is the first thing to check.

---

## Recreating the Model Files

The notebook saves the artifacts using:

```python
import joblib

joblib.dump(k_means, "k_means_model.pkl")
joblib.dump(sc, "scaler.pkl")
```

Make sure both generated files are committed to the GitHub repository.

---

## Model Details

The notebook trains the model using:

```python
from sklearn.cluster import KMeans

k_means = KMeans(n_clusters=6)
k_means.fit(scaled)
```

and scales the selected features using:

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
scaled = sc.fit_transform(df_train)
```

### Reproducibility note

The training notebook does not set `random_state` in `KMeans`. If the model is retrained, cluster IDs and cluster centers can change between training runs.

For future retraining, consider using a fixed seed, for example:

```python
k_means = KMeans(
    n_clusters=6,
    random_state=42,
    n_init="auto"
)
```

After retraining, save both the new model and the corresponding new scaler together.

---

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Joblib
- K-Means Clustering

---

## Files

### `app.py`
Streamlit user interface and prediction logic.

### `scaler.pkl`
Saved `StandardScaler` used during model training.

### `k_means_model.pkl`
Saved trained K-Means clustering model.

### `requirements.txt`
Python packages required for local execution and Streamlit Community Cloud.

### `Customer_segmentation_project.ipynb`
Notebook containing data preparation, exploratory analysis, feature engineering, model training, and model export.

---

## Future Improvements

Possible improvements include:

- adding business-friendly names to each cluster after profiling them,
- showing cluster visualizations,
- supporting batch prediction from CSV files,
- storing prediction history,
- retraining K-Means with `random_state` for reproducibility,
- adding automated model/version validation.

---

## License

Add the license of your choice before distributing the project publicly. A common option for portfolio projects is the MIT License.
