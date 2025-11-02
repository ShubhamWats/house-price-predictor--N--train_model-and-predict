import pandas as pd
import numpy as np
import joblib
from scipy import stats
from scipy.stats import randint


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans  
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def column_ratio(X):
    """
    Helper function to calculate a ratio between the first two columns of an array.
    We'll use this to create features like 'rooms_per_house'.
    """
    
    return X[:, [0]] / (X[:, [1]] + 1e-6)

def ratio_name(function_transformer, feature_names_in):
    """Helper function to name the new ratio feature."""
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),  
        FunctionTransformer(column_ratio, feature_names_out=ratio_name), 
        StandardScaler()  
    )


log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"), 
    StandardScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),  
    OneHotEncoder(handle_unknown="ignore")  
)


default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)


if __name__ == "__main__":
    
    try:
        housing_full = pd.read_csv(r"C:\Users\Shubham Wats\Desktop\trash 0.x\housing.csv\housing.csv")
        print("Successfully loaded housing.csv")
    except FileNotFoundError:
        print("Error: housing.csv not found.")
        print("Please check the file path in train_model.py")
        exit()
    except Exception as e:
        print(f"An error occurred loading the file: {e}")
        exit()

    housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                        labels=[1, 2, 3, 4, 5])

    strat_train_set, strat_test_set = train_test_split(
        housing_full, test_size=0.2, stratify=housing_full["income_cat"],
        random_state=42)

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    print("Data split into training and test sets.")

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    print("Fitting KMeans for geographic features...")
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(housing[["latitude", "longitude"]])
    geo_cluster_centers = kmeans.cluster_centers_

    def get_cluster_feature_names(transformer, feature_names_in):
        return [f"Cluster {i} similarity" for i in range(n_clusters)]

    geo_transformer = FunctionTransformer(
        rbf_kernel,
        kw_args=dict(Y=geo_cluster_centers, gamma=1.0),
        feature_names_out=get_cluster_feature_names
    )

    preprocessing = ColumnTransformer([
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                   "households", "median_income"]),
            
            ("geo", geo_transformer, ["latitude", "longitude"]),
            
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline
    )

    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),  
        ("random_forest", RandomForestRegressor(random_state=42)), 
    ])

    print("Starting model tuning (this may take a minute)...")
    
    gamma_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    geo_kw_args_list = [
        dict(Y=geo_cluster_centers, gamma=gamma_val) for gamma_val in gamma_values
    ]

    param_distribs = {
        'preprocessing__geo__kw_args': geo_kw_args_list,
        'random_forest__max_features': randint(low=2, high=20)
    }

    rnd_search = RandomizedSearchCV(
        full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
        scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1
    )
    
    rnd_search.fit(housing, housing_labels)

    print("Model tuning complete.")
    print("Best parameters found:")
    print(rnd_search.best_params_)

    final_model = rnd_search.best_estimator_

    print("\nEvaluating model on the unseen test set...")
    final_predictions = final_model.predict(X_test)
    final_rmse = root_mean_squared_error(y_test, final_predictions)
    print(f"\nFinal RMSE on test set: ${final_rmse:,.2f}")

    comparison_df = pd.DataFrame({
        "Actual Price": y_test.values[:100],
        "Predicted Price": final_predictions[:100]
    })
    comparison_df["Difference"] = comparison_df["Actual Price"] - comparison_df["Predicted Price"]
    print("\n-------------------------------------------------")
    print("Comparison of first 100 test predictions vs. actual prices:")
    print(comparison_df.to_string())
    print("-------------------------------------------------")
    
    model_filename = "my_california_housing_model.pkl"
    joblib.dump(final_model, model_filename)
    print(f"\nFinal model saved to {model_filename}")


