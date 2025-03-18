import numpy as np
import pandas as pd
import statsmodels.api as sm

def load_model_coefficients(filepath):
    """Loads the fitted model coefficients from a CSV file."""
    coefs_df = pd.read_csv(filepath)
    coefs = coefs_df.set_index("Variable")["Coefficient"].to_dict()
    return coefs

def compute_nested_logit_probabilities(products_df, coefs):
    """Computes choice probabilities using a Nested Logit Model (NLM)."""
    products_df = products_df.copy()
    products_df["const"] = 1.0

    # Compute utility scores
    utilities = (
        products_df["const"] * coefs.get("const", 0) +
        products_df["price"] * coefs.get("price", 0) +
        products_df["brand_strength"] * coefs.get("brand_strength", 0) +
        products_df["quality_score"] * coefs.get("quality_score", 0)
    )

    # Compute exponentiated utilities
    products_df["exp_utility"] = np.exp(utilities)

    # Compute group-level inclusive value (log-sum of exponentiated utilities)
    group_sum_utilities = products_df.groupby("group")["exp_utility"].sum()
    products_df["group_sum_util"] = products_df["group"].map(group_sum_utilities)

    # Compute probability of choosing a product given the group
    products_df["prob_within_group"] = products_df["exp_utility"] / products_df["group_sum_util"]

    # Compute probability of choosing the group
    total_sum_utilities = group_sum_utilities.sum()
    products_df["prob_group"] = products_df["group_sum_util"] / total_sum_utilities

    # Final probability of choosing each product (Nested Logit formula)
    products_df["predicted_choice_prob"] = products_df["prob_within_group"] * products_df["prob_group"]

    return products_df

# Example usage:
if __name__ == "__main__":
    # Load model coefficients
    coefs = load_model_coefficients("logit_coefficients.csv")

    # Define a sample set of products (including "group" column for nesting)
    sample_products = pd.DataFrame({
        "product_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "price": [5.0, 7.0, 6.5, 8.0, 4.5, 6.2, 9.1, 5.8],
        "brand_strength": [0.3, 0.6, 0.2, 0.8, 0.5, 0.4, 0.7, 0.3],
        "quality_score": [3.0, 4.5, 2.5, 4.0, 3.5, 3.2, 4.1, 2.8],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"]  # Two nests (A and B)
    })

    # Compute predicted choice probabilities using the Nested Logit Model
    predictions = compute_nested_logit_probabilities(sample_products, coefs)
    print(predictions[["product_id", "group", "predicted_choice_prob"]])

    # Save to file
    predictions.to_parquet("nested_predictions.parquet")
