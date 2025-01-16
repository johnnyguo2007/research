import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load the data
file_path = "/home/jguo/ftp/upload/Seeds_Fert.xlsx"
df = pd.read_excel(file_path)

# Print column names for debugging
print(df.columns)

# --- 1. Data Preparation ---
df['Germinated'] = np.where(df['num_of_seeds'].isna(), 0, 1)
df_fertilized = df[df['Fert_Mass'] > 0]
df_no_fertilizer = df[df['Fert_Mass'] == 0]

print("--- Data Preparation ---")
print(f"Total number of observations: {len(df)}")
print(f"Number of observations with fertilizer: {len(df_fertilized)}")
print(f"Number of observations without fertilizer: {len(df_no_fertilizer)}")
print("\n")

# --- 2. Germination Analysis ---
print("--- 2. Germination Analysis ---")

# Overall Germination Rate
overall_germination_rate = df['Germinated'].mean() * 100
print(f"Overall Germination Rate: {overall_germination_rate:.2f}%")

# Germination Rate by Group
germination_rate_fertilized = df_fertilized['Germinated'].mean() * 100
germination_rate_no_fertilizer = df_no_fertilizer['Germinated'].mean() * 100
print(f"Germination Rate (Fertilizer): {germination_rate_fertilized:.2f}%")
print(f"Germination Rate (No Fertilizer): {germination_rate_no_fertilizer:.2f}%")

# Chi-squared test for independence of fertilization and germination
observed = pd.crosstab(df['Fert_Mass'] > 0, df['Germinated'])
chi2, p, _, _ = stats.chi2_contingency(observed)
print(f"Chi-squared test for Germination vs. Fertilizer: Chi2 = {chi2:.2f}, p = {p:.3f}")

# Logistic Regression
if not df['Fert_Mass'].nunique() == 1: # Avoid error if only one unique value
    model_germination = smf.logit('Germinated ~ Fert_Mass', data=df).fit()
    print("\nLogistic Regression for Germination:")
    print(model_germination.summary())
else:
    print("\nSkipping Logistic Regression for Germination: Only one unique value in Fert_Mass.")

print("\n")

# --- 3. Seed Production Analysis (Among Germinated Plants) ---
df_germinated = df.dropna(subset=['num_of_seeds'])
df_germinated_fertilized = df_germinated[df_germinated['Fert_Mass'] > 0]
df_germinated_no_fertilizer = df_germinated[df_germinated['Fert_Mass'] == 0]

print("--- 3. Seed Production Analysis (Among Germinated Plants) ---")

print("\n--- Including germinated plants with zero seeds ---")
print("Seed Production (including zeros):")
print(f"Mean Seeds (Fertilizer): {df_germinated_fertilized['num_of_seeds'].mean():.2f}")
print(f"Mean Seeds (No Fertilizer): {df_germinated_no_fertilizer['num_of_seeds'].mean():.2f}")

# T-test or Mann-Whitney (checking for normality visually or with Shapiro-Wilk)
if len(df_germinated_fertilized) > 1 and len(df_germinated_no_fertilizer) > 1:
    shapiro_fertilized = stats.shapiro(df_germinated_fertilized['num_of_seeds'])
    shapiro_no_fertilizer = stats.shapiro(df_germinated_no_fertilizer['num_of_seeds'])
    print(f"Shapiro-Wilk test (Fertilizer): Statistic = {shapiro_fertilized.statistic:.3f}, p = {shapiro_fertilized.pvalue:.3f}")
    print(f"Shapiro-Wilk test (No Fertilizer): Statistic = {shapiro_no_fertilizer.statistic:.3f}, p = {shapiro_no_fertilizer.pvalue:.3f}")

    if shapiro_fertilized.pvalue > 0.05 and shapiro_no_fertilizer.pvalue > 0.05:
        ttest = stats.ttest_ind(df_germinated_fertilized['num_of_seeds'], df_germinated_no_fertilizer['num_of_seeds'], equal_var=False) # Assuming unequal variances
        print(f"T-test for Seed Production (including zeros): t = {ttest.statistic:.3f}, p = {ttest.pvalue:.3f}")
    else:
        mwu = stats.mannwhitneyu(df_germinated_fertilized['num_of_seeds'], df_germinated_no_fertilizer['num_of_seeds'], alternative='two-sided')
        print(f"Mann-Whitney U test for Seed Production (including zeros): U = {mwu.statistic:.3f}, p = {mwu.pvalue:.3f}")
else:
    print("Cannot perform statistical tests for seed production (including zeros) due to insufficient data in one or both groups.")

# Regression (including zeros)
if not df_germinated['Fert_Mass'].nunique() == 1:
    model_seeds_incl_zero_linear = smf.ols('num_of_seeds ~ Fert_Mass', data=df_germinated).fit()
    print("\nLinear Regression for Seed Production (including zeros):")
    print(model_seeds_incl_zero_linear.summary())
else:
    print("\nSkipping Regression for Seed Production (including zeros): Only one unique value in Fert_Mass among germinated plants.")

print("\n--- Excluding germinated plants with zero seeds ---")
df_germinated_positive_seeds = df_germinated[df_germinated['num_of_seeds'] > 0]
df_germinated_positive_seeds_fertilized = df_germinated_positive_seeds[df_germinated_positive_seeds['Fert_Mass'] > 0]
df_germinated_positive_seeds_no_fertilizer = df_germinated_positive_seeds[df_germinated_positive_seeds['Fert_Mass'] == 0]

print("Seed Production (excluding zeros):")
if not df_germinated_positive_seeds_fertilized.empty:
    print(f"Mean Seeds (Fertilizer): {df_germinated_positive_seeds_fertilized['num_of_seeds'].mean():.2f}")
if not df_germinated_positive_seeds_no_fertilizer.empty:
    print(f"Mean Seeds (No Fertilizer): {df_germinated_positive_seeds_no_fertilizer['num_of_seeds'].mean():.2f}")

# T-test or Mann-Whitney (checking for normality)
if len(df_germinated_positive_seeds_fertilized) > 1 and len(df_germinated_positive_seeds_no_fertilizer) > 1:
    shapiro_pos_fertilized = stats.shapiro(df_germinated_positive_seeds_fertilized['num_of_seeds'])
    shapiro_pos_no_fertilizer = stats.shapiro(df_germinated_positive_seeds_no_fertilizer['num_of_seeds'])
    print(f"Shapiro-Wilk test (Fertilizer, positive seeds): Statistic = {shapiro_pos_fertilized.statistic:.3f}, p = {shapiro_pos_fertilized.pvalue:.3f}")
    print(f"Shapiro-Wilk test (No Fertilizer, positive seeds): Statistic = {shapiro_pos_no_fertilizer.statistic:.3f}, p = {shapiro_pos_no_fertilizer.pvalue:.3f}")

    if shapiro_pos_fertilized.pvalue > 0.05 and shapiro_pos_no_fertilizer.pvalue > 0.05:
        ttest_pos = stats.ttest_ind(df_germinated_positive_seeds_fertilized['num_of_seeds'], df_germinated_positive_seeds_no_fertilizer['num_of_seeds'], equal_var=False)
        print(f"T-test for Seed Production (excluding zeros): t = {ttest_pos.statistic:.3f}, p = {ttest_pos.pvalue:.3f}")
    else:
        mwu_pos = stats.mannwhitneyu(df_germinated_positive_seeds_fertilized['num_of_seeds'], df_germinated_positive_seeds_no_fertilizer['num_of_seeds'], alternative='two-sided')
        print(f"Mann-Whitney U test for Seed Production (excluding zeros): U = {mwu_pos.statistic:.3f}, p = {mwu_pos.pvalue:.3f}")
elif not df_germinated_positive_seeds_fertilized.empty and not df_germinated_positive_seeds_no_fertilizer.empty:
    print("Cannot perform statistical tests for seed production (excluding zeros) due to insufficient data in one or both groups.")
else:
    print("Cannot perform statistical tests for seed production (excluding zeros) as one or both groups are empty.")

# Regression (excluding zeros)
if not df_germinated_positive_seeds['Fert_Mass'].nunique() == 1 and len(df_germinated_positive_seeds) > 0:
    model_seeds_excl_zero_linear = smf.ols('num_of_seeds ~ Fert_Mass', data=df_germinated_positive_seeds).fit()
    print("\nLinear Regression for Seed Production (excluding zeros):")
    print(model_seeds_excl_zero_linear.summary())
else:
    print("\nSkipping Regression for Seed Production (excluding zeros): Either only one unique value in Fert_Mass or no data.")

print("\n--- 4. Overall Comparison and Conclusions ---")
print("Based on the analysis:")
print(f"- Germination rate with fertilizer: {germination_rate_fertilized:.2f}% vs. without: {germination_rate_no_fertilizer:.2f}%.")
if p < 0.05:
    print("- The difference in germination rates between the groups is statistically significant (p < 0.05).")
else:
    print("- There is no statistically significant difference in germination rates between the groups.")

if len(df_germinated_fertilized) > 0 and len(df_germinated_no_fertilizer) > 0:
    mean_seeds_fertilized_incl_zero = df_germinated_fertilized['num_of_seeds'].mean()
    mean_seeds_no_fertilizer_incl_zero = df_germinated_no_fertilizer['num_of_seeds'].mean()
    print(f"- Mean seed production (including zeros) with fertilizer: {mean_seeds_fertilized_incl_zero:.2f} vs. without: {mean_seeds_no_fertilizer_incl_zero:.2f}.")

if len(df_germinated_positive_seeds_fertilized) > 0 and len(df_germinated_positive_seeds_no_fertilizer) > 0:
    mean_seeds_fertilized_excl_zero = df_germinated_positive_seeds_fertilized['num_of_seeds'].mean()
    mean_seeds_no_fertilizer_excl_zero = df_germinated_positive_seeds_no_fertilizer['num_of_seeds'].mean()
    print(f"- Mean seed production (excluding zeros) with fertilizer: {mean_seeds_fertilized_excl_zero:.2f} vs. without: {mean_seeds_no_fertilizer_excl_zero:.2f}.")

print("\nFurther analysis of regression models can provide insights into the relationship between fertilizer mass and seed production.")