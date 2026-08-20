import numpy as np
import pandas as pd


def rank_cfg_scale_grid(csv_path_or_buffer):
    # 1. Load dataset
    df = pd.read_csv(csv_path_or_buffer)

    # 2. Define GRAVY Gaussian solubility score (target = -0.30, sigma = 0.15)
    g_target = -0.30
    sigma_g = 0.15
    df["s_gravy"] = np.exp(-((df["mean_gravy_score"] - g_target) ** 2) / (2 * (sigma_g**2)))

    # 3. Min-Max normalize linear metrics
    metrics_to_normalize = [
        "mean_charge_score",
        "mean_alpha_score",
        "mean_beta_score",
        "mean_boman_index",
        "success_rate",
        "novelty_rate",
        "repeat_rate",
    ]

    for col in metrics_to_normalize:
        min_val = df[col].min()
        max_val = df[col].max()
        norm_col_name = f"norm_{col}"
        df[norm_col_name] = (df[col] - min_val) / (max_val - min_val)

    # Invert repeat rate (lower repeat_rate -> higher score)
    df["norm_repeat_rate_inv"] = 1.0 - df["norm_repeat_rate"]

    # 4. Define Weight Budgets:
    # Primary Biophysical Metrics = 90% (18% each or weighted by structural focus)
    # Secondary Generation Metrics = 10% (4% success_rate, 3% novelty_rate, 3% inverted repeat_rate)
    w_succ = 0.04
    w_nov = 0.03
    w_rep = 0.03
    secondary_utility = (
        w_succ * df["norm_success_rate"]
        + w_nov * df["norm_novelty_rate"]
        + w_rep * df["norm_repeat_rate_inv"]
    )

    # A. Balanced Profile
    df["utility_balanced"] = (
        0.18 * df["norm_mean_charge_score"]
        + 0.18 * df["norm_mean_alpha_score"]
        + 0.18 * df["norm_mean_beta_score"]
        + 0.18 * df["norm_mean_boman_index"]
        + 0.18 * df["s_gravy"]
        + secondary_utility
    )

    # B. Alpha Focus Profile
    df["utility_alpha_focus"] = (
        0.18 * df["norm_mean_charge_score"]
        + 0.36 * df["norm_mean_alpha_score"]
        + 0.00 * df["norm_mean_beta_score"]
        + 0.18 * df["norm_mean_boman_index"]
        + 0.18 * df["s_gravy"]
        + secondary_utility
    )

    # C. Beta Focus Profile
    df["utility_beta_focus"] = (
        0.18 * df["norm_mean_charge_score"]
        + 0.00 * df["norm_mean_alpha_score"]
        + 0.36 * df["norm_mean_beta_score"]
        + 0.18 * df["norm_mean_boman_index"]
        + 0.18 * df["s_gravy"]
        + secondary_utility
    )

    # 5. Extract top parameter combinations with ALL metrics
    display_cols = [
        "species_scale",
        "groups_scale",
        "mic_scale",
        "mean_charge_score",
        "mean_alpha_score",
        "mean_beta_score",
        "mean_boman_index",
        "mean_gravy_score",
        "success_rate",
        "novelty_rate",
        "repeat_rate",
    ]

    top_balanced = df.sort_values(by="utility_balanced", ascending=False).head(5)
    top_alpha = df.sort_values(by="utility_alpha_focus", ascending=False).head(5)
    top_beta = df.sort_values(by="utility_beta_focus", ascending=False).head(5)

    return {
        "balanced": top_balanced[display_cols + ["utility_balanced"]],
        "alpha_focus": top_alpha[display_cols + ["utility_alpha_focus"]],
        "beta_focus": top_beta[display_cols + ["utility_beta_focus"]],
    }


# Execute
results = rank_cfg_scale_grid("./output/20251224-123317/grid_search_scales/grid_search_summary.csv")
print("Top Balanced Scales:\n", results["balanced"][['species_scale', 'groups_scale', 'mic_scale', 'mean_charge_score', 'mean_alpha_score', 'mean_beta_score', 'mean_boman_index', 'mean_gravy_score', 'utility_balanced']])
print("Top alpha_focus Scales:\n", results["alpha_focus"][['species_scale', 'groups_scale', 'mic_scale', 'mean_charge_score', 'mean_alpha_score', 'mean_beta_score', 'mean_boman_index', 'mean_gravy_score', 'utility_alpha_focus']])
print("Top beta_focus Scales:\n", results["beta_focus"][['species_scale', 'groups_scale', 'mic_scale', 'mean_charge_score', 'mean_alpha_score', 'mean_beta_score', 'mean_boman_index', 'mean_gravy_score', 'utility_beta_focus']])