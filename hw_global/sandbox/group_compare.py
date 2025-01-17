import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

# Load the data
file_path = "/home/jguo/ftp/upload/Seeds_Fert.xlsx"
output_dir = os.path.dirname(file_path)
df = pd.read_excel(file_path)

# Print column names for debugging
print(df.columns)

# --- 1. Data Preparation ---
df['Germinated'] = np.where(df['num_of_seeds'].isna(), 0, 1)
df_fertilized = df[df['Fert_Mass'] > 0].copy()
df_no_fertilizer = df[df['Fert_Mass'] == 0].copy()

print("--- 1. Data Preparation ---")
print(f"Total number of observations: {len(df)}")
print(f"Number of observations with fertilizer: {len(df_fertilized)}")
print(f"Number of observations without fertilizer: {len(df_no_fertilizer)}")
print("\n")

# --- 2. Overall Impact of Fertilizer on Seed Production ---
print("\n--- 2. Overall Impact of Fertilizer on Seed Production ---")

# Germination Rate by Group
germination_rate_fertilized = df_fertilized['Germinated'].mean() * 100
germination_rate_no_fertilizer = df_no_fertilizer['Germinated'].mean() * 100

# Seed Production (including zeros)
mean_seeds_fertilized_incl_zero = df_fertilized[df_fertilized['Germinated']==1]['num_of_seeds'].mean()
mean_seeds_no_fertilizer_incl_zero = df_no_fertilizer[df_no_fertilizer['Germinated']==1]['num_of_seeds'].mean()

# Calculate expected seed production per plant
expected_seeds_fertilized = germination_rate_fertilized / 100 * mean_seeds_fertilized_incl_zero
expected_seeds_no_fertilizer = germination_rate_no_fertilizer / 100 * mean_seeds_no_fertilizer_incl_zero

print(f"Expected seeds per plant (Fertilizer): {expected_seeds_fertilized:.2f}")
print(f"Expected seeds per plant (No Fertilizer): {expected_seeds_no_fertilizer:.2f}")

# Statistical Tests for Expected Seed Production
# Create temporary variables for calculation
df_fertilized.loc[:, 'temp_expected_seeds'] = df_fertilized['Germinated'] * df_fertilized['num_of_seeds'].fillna(0)
df_no_fertilizer.loc[:, 'temp_expected_seeds'] = df_no_fertilizer['Germinated'] * df_no_fertilizer['num_of_seeds'].fillna(0)

# Initialize variables for test results
ttest_expected, mwu_expected = None, None

# Statistical test for expected seed production
if len(df_fertilized) > 1 and len(df_no_fertilizer) > 1:
    if stats.shapiro(df_fertilized['temp_expected_seeds']).pvalue > 0.05 and stats.shapiro(df_no_fertilizer['temp_expected_seeds']).pvalue > 0.05:
        ttest_expected = stats.ttest_ind(
            df_fertilized['temp_expected_seeds'],
            df_no_fertilizer['temp_expected_seeds'],
            equal_var=False
        )
        print(f"T-test for expected seed production: t = {ttest_expected.statistic:.3f}, p = {ttest_expected.pvalue:.3f}")
    else:
        mwu_expected = stats.mannwhitneyu(
            df_fertilized['temp_expected_seeds'],
            df_no_fertilizer['temp_expected_seeds'],
            alternative='two-sided'
        )
        print(f"Mann-Whitney U test for expected seed production: U = {mwu_expected.statistic:.3f}, p = {mwu_expected.pvalue:.3f}")
else:
    print("Cannot perform statistical tests for expected seed production due to insufficient data.")

# --- 3. Germination Analysis ---
print("\n--- 3. Germination Analysis ---")

# Overall Germination Rate
overall_germination_rate = df['Germinated'].mean() * 100
print(f"Overall Germination Rate: {overall_germination_rate:.2f}%")

print(f"Germination Rate (Fertilizer): {germination_rate_fertilized:.2f}%")
print(f"Germination Rate (No Fertilizer): {germination_rate_no_fertilizer:.2f}%")

# Chi-squared test for independence of fertilization and germination
observed = pd.crosstab(df['Fert_Mass'] > 0, df['Germinated'])
chi2, p, _, _ = stats.chi2_contingency(observed)
print(f"Chi-squared test for Germination vs. Fertilizer: Chi2 = {chi2:.2f}, p = {p:.3f}")

# Logistic Regression
if not df['Fert_Mass'].nunique() == 1:
    model_germination = smf.logit('Germinated ~ Fert_Mass', data=df).fit()
    print("\nLogistic Regression for Germination:")
    print(model_germination.summary())

    # Output paragraph for Logistic Regression
    print("\n--- Logistic Regression Results ---")
    print("The logistic regression model examines the relationship between fertilizer mass and the probability of germination.")
    print(f"The model's pseudo R-squared value is {model_germination.prsquared:.3f}, indicating the proportion of variance in the outcome variable explained by the model.")
    print("The coefficients in the model represent the change in the log-odds of germination associated with a one-unit increase in fertilizer mass.")
    print(f"In this case, the coefficient for Fert_Mass is {model_germination.params['Fert_Mass']:.3f} (p = {model_germination.pvalues['Fert_Mass']:.3f}).")
    if model_germination.pvalues['Fert_Mass'] < 0.05:
        print("This indicates that fertilizer mass has a statistically significant effect on the odds of germination.")
        if model_germination.params['Fert_Mass'] > 0:
            print("Specifically, an increase in fertilizer mass is associated with an increase in the odds of germination.")
        else:
            print("Specifically, an increase in fertilizer mass is associated with a decrease in the odds of germination.")
    else:
        print("This indicates that fertilizer mass does not have a statistically significant effect on the odds of germination.")

    # Explanation of the logistic regression plot
    print("\n--- Logistic Regression Plot Explanation ---")
    print("The logistic regression plot visualizes the relationship between fertilizer mass and the probability of germination.")
    print("The x-axis represents the fertilizer mass (in grams), and the y-axis represents the predicted probability of germination.")
    print("The blue dots represent the actual data points, where each dot corresponds to an observation in the dataset.")
    print("The red curve represents the predicted probability of germination based on the logistic regression model.")
    print("As the red curve moves from left to right along the x-axis (increasing fertilizer mass), the corresponding y-values show how the model estimates the probability of germination changes.")
    print("The shape of the curve illustrates the non-linear relationship between fertilizer mass and the probability of germination that is characteristic of logistic regression.")

    # Plot for Logistic Regression
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Fert_Mass'], df['Germinated'], color='skyblue', label='Data Points')

    # Generate x-values for prediction
    x_pred = np.linspace(df['Fert_Mass'].min(), df['Fert_Mass'].max(), 100)

    # Predict probabilities
    y_pred = model_germination.predict(pd.DataFrame({'Fert_Mass': x_pred}))

    # Plot logistic curve
    plt.plot(x_pred, y_pred, color='red', label='Logistic Regression Curve')

    plt.xlabel('Fertilizer Mass (g)')
    plt.ylabel('Probability of Germination')
    plt.title('Logistic Regression: Germination vs. Fertilizer Mass')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'logistic_regression_germination.png'), bbox_inches='tight', dpi=300)
    plt.close()

else:
    print("\nSkipping Logistic Regression for Germination: Only one unique value in Fert_Mass.")

# Summary Paragraph for Germination Analysis
print("\n--- Germination Analysis Summary ---")
print(f"The overall germination rate across all plants was {overall_germination_rate:.2f}%.")
print(f"When comparing plants that received fertilizer to those that did not, the germination rates were {germination_rate_fertilized:.2f}% and {germination_rate_no_fertilizer:.2f}%, respectively.")
if p < 0.05:
    print(f"A chi-squared test indicated that this difference in germination rates is statistically significant (Chi2 = {chi2:.2f}, p = {p:.3f}).")
    print("This suggests that the application of fertilizer has a statistically significant association with the likelihood of germination.")
else:
    print(f"A chi-squared test indicated that the difference in germination rates is not statistically significant (Chi2 = {chi2:.2f}, p = {p:.3f}).")
    print("This suggests that the application of fertilizer does not have a statistically significant association with the likelihood of germination.")

# Plot for Germination Analysis
plt.figure(figsize=(8, 6))
plt.bar(["No Fertilizer", "Fertilizer"], [germination_rate_no_fertilizer, germination_rate_fertilized], color=['skyblue', 'lightgreen'])
plt.ylabel("Germination Rate (%)")
plt.title("Comparison of Germination Rates")
for i, v in enumerate([germination_rate_no_fertilizer, germination_rate_fertilized]):
    plt.text(i, v + 2, f"{v:.2f}%", ha='center', va='bottom')
plt.ylim(0, 100)
plt.savefig(os.path.join(output_dir, 'germination_rates.png'), bbox_inches='tight', dpi=300)
plt.close()

print("\n")

# --- 4. Seed Production Analysis (Among Germinated Plants) ---
df_germinated = df.dropna(subset=['num_of_seeds'])
df_germinated_fertilized = df_germinated[df_germinated['Fert_Mass'] > 0]
df_germinated_no_fertilizer = df_germinated[df_germinated['Fert_Mass'] == 0]

print("--- 4. Seed Production Analysis (Among Germinated Plants) ---")

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
if not df_germinated_fertilized['Fert_Mass'].nunique() == 1:
    model_seeds_incl_zero_linear = smf.ols('num_of_seeds ~ Fert_Mass', data=df_germinated_fertilized).fit()
    print("\nLinear Regression for Seed Production (including zeros, fertilized group only):")
    print(model_seeds_incl_zero_linear.summary())
    
    # Plot for Linear Regression (including zeros) with Confidence Band
    plt.figure(figsize=(8, 6))
    plt.scatter(df_germinated_fertilized['Fert_Mass'], df_germinated_fertilized['num_of_seeds'], label='Data Points')

    # Generate x-values for prediction
    x_pred = np.linspace(df_germinated_fertilized['Fert_Mass'].min(), df_germinated_fertilized['Fert_Mass'].max(), 50)
    
    # Predict y-values and get confidence interval
    pred_obj = model_seeds_incl_zero_linear.get_prediction(pd.DataFrame({'Fert_Mass': x_pred}))
    y_pred = pred_obj.predicted_mean
    conf_int = pred_obj.conf_int(alpha=0.05)  # 95% confidence interval

    # Plot regression line
    plt.plot(x_pred, y_pred, color='red', label=f'Regression Line: y = {model_seeds_incl_zero_linear.params.iloc[1]:.2f}x + {model_seeds_incl_zero_linear.params.iloc[0]:.2f}')
    
    # Plot confidence band
    plt.fill_between(x_pred, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.3, label='95% Confidence Band')

    plt.xlabel('Fertilizer Mass (g)')
    plt.ylabel('Number of Seeds')
    plt.title('Linear Regression for Seed Production\n(Including Zeros, Fertilized Group Only)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'regression_including_zeros.png'), bbox_inches='tight', dpi=300)
    plt.close()
else:
    print("\nSkipping Regression for Seed Production (including zeros): Only one unique value in Fert_Mass among fertilized germinated plants.")

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
if not df_germinated_positive_seeds_fertilized['Fert_Mass'].nunique() == 1 and len(df_germinated_positive_seeds_fertilized) > 0:
    model_seeds_excl_zero_linear = smf.ols('num_of_seeds ~ Fert_Mass', data=df_germinated_positive_seeds_fertilized).fit()
    print("\nLinear Regression for Seed Production (excluding zeros, fertilized group only):")
    print(model_seeds_excl_zero_linear.summary())
    
    # Plot for Linear Regression (excluding zeros) with Confidence Band
    plt.figure(figsize=(8, 6))
    plt.scatter(df_germinated_positive_seeds_fertilized['Fert_Mass'], df_germinated_positive_seeds_fertilized['num_of_seeds'], label='Data Points')
    
    # Generate x-values for prediction
    x_pred_excl_zero = np.linspace(df_germinated_positive_seeds_fertilized['Fert_Mass'].min(), df_germinated_positive_seeds_fertilized['Fert_Mass'].max(), 50)
    
    # Predict y-values and get confidence interval
    pred_obj_excl_zero = model_seeds_excl_zero_linear.get_prediction(pd.DataFrame({'Fert_Mass': x_pred_excl_zero}))
    y_pred_excl_zero = pred_obj_excl_zero.predicted_mean
    conf_int_excl_zero = pred_obj_excl_zero.conf_int(alpha=0.05)

    # Plot regression line
    plt.plot(x_pred_excl_zero, y_pred_excl_zero, color='red', label=f'Regression Line: y = {model_seeds_excl_zero_linear.params.iloc[1]:.2f}x + {model_seeds_excl_zero_linear.params.iloc[0]:.2f}')
    
    # Plot confidence band
    plt.fill_between(x_pred_excl_zero, conf_int_excl_zero[:, 0], conf_int_excl_zero[:, 1], color='gray', alpha=0.3, label='95% Confidence Band')

    plt.xlabel('Fertilizer Mass (g)')
    plt.ylabel('Number of Seeds')
    plt.title('Linear Regression for Seed Production\n(Excluding Zeros, Fertilized Group Only)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'regression_excluding_zeros.png'), bbox_inches='tight', dpi=300)
    plt.close()
else:
    print("\nSkipping Regression for Seed Production (excluding zeros): Either only one unique value in Fert_Mass or no data among fertilized plants.")

# --- 5. Overall Summary and Conclusions ---
print("\n--- 5. Overall Summary and Conclusions ---")
print("This study investigated the impact of fertilizer on seed production, considering both the effect on germination and the number of seeds produced by germinated plants.")

print("\n**Overall Impact:**")
print(f"- When both germination rates and seed production are considered, the expected number of seeds per plant with fertilizer was {expected_seeds_fertilized:.2f} and without fertilizer was {expected_seeds_no_fertilizer:.2f}.")

if ttest_expected is not None:
    if ttest_expected.pvalue < 0.05:
        print("- A statistical test indicated a significant difference in the overall expected seed production per plant between the fertilizer and no-fertilizer groups.")
        if expected_seeds_fertilized > expected_seeds_no_fertilizer:
            print("  - This suggests that despite any impact on germination, the increase in seed production among germinated plants may compensate for the loss, potentially leading to a net positive effect of fertilizer on overall seed yield.")
        else:
            print("  - This suggests that the impact on germination may outweigh the increase in seed production, potentially leading to a net negative effect of fertilizer on overall seed yield.")
    else:
        print("- A statistical test found no significant difference in the overall expected seed production per plant between the fertilizer and no-fertilizer groups.")
elif mwu_expected is not None:
    if mwu_expected.pvalue < 0.05:
        print("- A statistical test indicated a significant difference in the overall expected seed production per plant between the fertilizer and no-fertilizer groups.")
        if expected_seeds_fertilized > expected_seeds_no_fertilizer:
            print("  - This suggests that despite any impact on germination, the increase in seed production among germinated plants may compensate for the loss, potentially leading to a net positive effect of fertilizer on overall seed yield.")
        else:
            print("  - This suggests that the impact on germination may outweigh the increase in seed production, potentially leading to a net negative effect of fertilizer on overall seed yield.")
    else:
        print("- A statistical test found no significant difference in the overall expected seed production per plant between the fertilizer and no-fertilizer groups.")

print("\n**Germination:**")
print(f"- The overall germination rate was {overall_germination_rate:.2f}%. Plants treated with fertilizer had a germination rate of {germination_rate_fertilized:.2f}%, while those without fertilizer had a rate of {germination_rate_no_fertilizer:.2f}%.")
if p < 0.05:
    print(f"- A chi-squared test revealed a statistically significant association between fertilizer use and germination (Chi2 = {chi2:.2f}, p = {p:.3f}).")
    if germination_rate_fertilized > germination_rate_no_fertilizer:
      print("- This suggests that fertilizer use is associated with a higher likelihood of germination.")
    else:
      print("- This suggests that fertilizer use is associated with a lower likelihood of germination.")
else:
    print("- A chi-squared test found no statistically significant association between fertilizer use and germination.")

print("\n- The logistic regression model further explored the relationship between fertilizer mass and the probability of germination, showing how the odds of germination change with increasing fertilizer.")

print("\n**Seed Production (Among Germinated Plants):**")
if len(df_germinated_fertilized) > 0 and len(df_germinated_no_fertilizer) > 0:
    print(f"- Among germinated plants, those treated with fertilizer produced an average of {mean_seeds_fertilized_incl_zero:.2f} seeds, while those without fertilizer produced {mean_seeds_no_fertilizer_incl_zero:.2f} seeds.")
    if len(df_germinated_fertilized) > 1 and len(df_germinated_no_fertilizer) > 1:
        if shapiro_fertilized.pvalue > 0.05 and shapiro_no_fertilizer.pvalue > 0.05:
            if ttest.pvalue < 0.05:
                print(f"  - A t-test indicated a statistically significant difference in seed production between the two groups (t = {ttest.statistic:.3f}, p = {ttest.pvalue:.3f}).")
            else:
                print(f"  - A t-test did not find a statistically significant difference in seed production between the two groups (t = {ttest.statistic:.3f}, p = {ttest.pvalue:.3f}).")
        else:
            if mwu.pvalue < 0.05:
                print(f"  - A Mann-Whitney U test indicated a statistically significant difference in seed production between the two groups (U = {mwu.statistic:.3f}, p = {mwu.pvalue:.3f}).")
            else:
                print(f"  - A Mann-Whitney U test did not find a statistically significant difference in seed production between the two groups (U = {mwu.statistic:.3f}, p = {mwu.pvalue:.3f}).")

print("\n- The linear regression models further investigated the relationship between fertilizer mass and the number of seeds produced. These models help to quantify the effect of fertilizer mass on seed production.")

print("\n**Conclusion:**")
print("In conclusion, this study reveals a complex relationship between fertilizer use and seed production. The analysis considers both germination rates and seed production to assess the overall impact of fertilizer. The analysis of expected seed production per plant provides a way to assess the net effect of fertilizer. Further research could investigate the optimal fertilizer levels to maximize seed production while considering the impact on germination.")