import numpy as np
import pandas as pd
import joblib
import tarfile
import urllib.request
from pathlib import Path
from scipy import stats
from scipy.stats import randint

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

## 1. Get Data
def load_housing_data():
    """Fetches and loads the California Housing dataset."""
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

## 2. Custom Transformers & Helper Functions

# Helper function for ratio pipeline
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

# Helper function for ratio pipeline
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add features based on geographic cluster similarity.
    Uses KMeans to find clusters and rbf_kernel to measure similarity.
    """
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

## 3. Preprocessing Pipelines

# Pipeline for ratio features (e.g., bedrooms_per_room)
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

# Pipeline for log-transformed features
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

# Pipeline for categorical features
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

# Pipeline for default numerical features
default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

# Custom cluster similarity transformer instance
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

# --- The Main Preprocessing ColumnTransformer ---
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # Apply default_num_pipeline to remaining numeric cols

## 4. Main Execution
if __name__ == "__main__":
    
    print("Loading data...")
    housing_full = load_housing_data()

    # --- Create Stratified Test/Train Split ---
    # Create the income category attribute for stratified sampling
    housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                        labels=[1, 2, 3, 4, 5])

    # Perform the stratified split
    strat_train_set, strat_test_set = train_test_split(
        housing_full, test_size=0.2, stratify=housing_full["income_cat"],
        random_state=42)

    # Drop the income_cat column from both sets
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    print("Data split complete.")

    # --- Separate Predictors and Labels ---
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    # --- Define Full Pipeline (Preprocessing + Model) ---
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ])

    # --- Fine-Tune Model with RandomizedSearchCV ---
    print("Starting model tuning with RandomizedSearchCV...")
    
    # Define the parameter distribution for the search
    param_distribs = {
        'preprocessing__geo__n_clusters': randint(low=3, high=50),
        'random_forest__max_features': randint(low=2, high=20)
    }

    # Initialize and run the search
    rnd_search = RandomizedSearchCV(
        full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
        scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1
    )
    
    rnd_search.fit(housing, housing_labels)

    print("Model tuning complete.")
    print("Best parameters found:")
    print(rnd_search.best_params_)

    # Get the best model
    final_model = rnd_search.best_estimator_

    # --- Analyze Final Model (Feature Importances) ---
    print("\nFeature importances:")
    try:
        feature_importances = final_model["random_forest"].feature_importances_
        feature_names = final_model["preprocessing"].get_feature_names_out()
        
        for importance, name in sorted(zip(feature_importances, feature_names), reverse=True):
            print(f"{name}: {importance:.4f}")
            
    except Exception as e:
        print(f"Could not retrieve feature importances: {e}")


    # --- Evaluate Final Model on the Test Set ---
    print("\nEvaluating model on the test set...")
    
    final_predictions = final_model.predict(X_test)
    final_rmse = root_mean_squared_error(y_test, final_predictions)
    
    print(f"\nFinal RMSE on test set: {final_rmse:.2f}")

    # Compute 95% confidence interval for the RMSE
    squared_errors = (final_predictions - y_test) ** 2
    confidence = 0.95
    
    boot_result = stats.bootstrap([squared_errors], 
                                  lambda se: np.sqrt(np.mean(se)),
                                  confidence_level=confidence, 
                                  random_state=42)
    
    rmse_lower, rmse_upper = boot_result.confidence_interval
    print(f"95% confidence interval for test RMSE: [{rmse_lower:.2f}, {rmse_upper:.2f}]")

    # --- Save Final Model ---
    model_filename = "my_california_housing_model.pkl"
    joblib.dump(final_model, model_filename)
    print(f"\nFinal model saved to {model_filename}")