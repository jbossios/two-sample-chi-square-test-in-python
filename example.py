
from scipy.stats import chi2, norm, chi2_contingency
import numpy as np
import pandas as pd
import math


def generate_data(
        sample_size,
        conversion_rate_A,  # conversion rate for group A
        conversion_rate_B  # conversion rate for group B
    ):
    """Generate fake data to perform a two-sample chi-square test"""

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate data for group A and B
    group_A_converted = np.random.binomial(1, conversion_rate_A, sample_size)
    group_A_not_converted = 1 - group_A_converted
    group_B_converted = np.random.binomial(1, conversion_rate_B, sample_size)
    group_B_not_converted = 1 - group_B_converted

    # Create a DataFrame to store the data
    data = pd.DataFrame({
        'Group': ['A'] * sample_size + ['B'] * sample_size,
        'Converted': np.concatenate([group_A_converted, group_B_converted]),
        'Not Converted': np.concatenate([group_A_not_converted, group_B_not_converted])
    })

    # Create a contingency table
    contingency_table = pd.crosstab(data['Group'], data['Converted'])

    return contingency_table


def get_min_sample_size(
        p1,  # conversion rate for group A
        des,  # desired effect size
        alpha = 0.05,  # significance level
        power = 0.8  # statistical power
    ):
    """
    Estimate minimum sample size for chi-square test
    Assumption: sigma_A = sigma_B (using pooled probability)
    """
    # Find Z_beta from desired power
    Z_beta = norm.ppf(power)  # ppf = Percent Point Function (inverse of Cumulative Distribution Function)

    # Find Z_alpha
    Z_alpha = norm.ppf(1 - alpha / 2)

    # Estimate minimum sample size
    p2 = p1 + des
    avgp = 0.5 * (p1 + p2)  # pooled proportions
    var = avgp * (1 - avgp)  # variance
    return math.ceil(2 * var * (Z_alpha + Z_beta)**2 / des**2)


def main() -> None:
    # Define parameters
    alpha = 0.05  # i.e. 95% CL
    power = 0.8  # beta = 0.2
    des = 0.006  # 3% effect on nominal conversion rate (0.2)

    # Estimate minimum sample size
    min_sample_size = get_min_sample_size(p1 = 0.2, des = des, alpha = alpha, power = power)
    print(f'Minimum sample size for alpha = {alpha}, power = {power} and des = {des}: {min_sample_size}')

    # Test #1 (rejecting the null hypothesis)
    print('\n>>> Test #1: rejecting the null hypothesis <<<')

    # Generate fake data
    data = generate_data(sample_size = min_sample_size, conversion_rate_A = 0.2, conversion_rate_B = 0.21)
    print("Contingency Table:")
    print(data)

    # Perform the chi-squared test
    stat, pvalue, _, _ = chi2_contingency(data, correction = False)  # correction = False (none of the expected counts is smaller than 5)
    if pvalue < alpha:
        print(f"Decision: There is a significant difference between the groups (p-value = {pvalue}, chi2-statistics = {round(stat, 2)}).")
    else:
        print(f"Decision: There is no significant difference between the groups (p-value = {pvalue}, chi2-statistics = {round(stat, 2)}).")

    # Let's calculate the chi2-statistics by hand
    dof = 1  # degrees of freedom = (r-1)(c-1), where r=#rows and c=#columns
    # If the null hypothesis is true, then there would be no significant difference b/w groups A and B
    # Under that assumption, we can use the observed values to calculate the expected ratio of conversions
    expected_proportion = (data[1]["A"] + data[1]["B"]) / (data[1]["A"] + data[1]["B"] + data[0]["A"] + data[0]["B"])
    expected_1 = expected_proportion * min_sample_size
    expected_0 = min_sample_size - expected_1
    obs = np.asarray(data)  # observed values
    exp = np.array([[expected_0, expected_1], [expected_0, expected_1]])  # expected values
    terms = (obs - exp) ** 2 / exp
    my_stat = terms.sum(axis = None)
    print(f'chi2-statistics calculated by hand = {round(my_stat, 2)}')
   
    # Let's calculate the critical chi2-statistics
    critical_chi2_stat = round(chi2.ppf(1 - alpha, dof), 2)  # ppf = percent point function (inverse of the cumulative distribution function)
    print(f'{critical_chi2_stat = }')
    # note: we need to get a chi2-statistics higher than this critical chi2-statistics value to reject the null hypothesis
    
    my_pvalue = chi2.sf(my_stat, dof)  # 1 - cdf, where cdf = cumulative distribution function = P(X <= x) = probability that X will have a value <= x
    print(f'p-value calculated by hand = {my_pvalue}') 


    # Test #2 (failing to reject the null hypothesis)
    print('\n>>> Test #2: failing to reject the null hypothesis <<<')

    # Generate fake data
    data = generate_data(sample_size = min_sample_size, conversion_rate_A = 0.2, conversion_rate_B = 0.202)
    print("Contingency Table:")
    print(data)

    # Perform the chi-squared test
    stat, pvalue, _, _ = chi2_contingency(data, correction = False)  # correction = False (none of the expected counts is smaller than 5)
    if pvalue < alpha:
        print(f"Decision: There is a significant difference between the groups (p-value = {round(pvalue, 3)}, chi2-statistics = {round(stat, 2)}).")
    else:
        print(f"Decision: There is no significant difference between the groups (p-value = {round(pvalue, 3)}, chi2-statistics = {round(stat, 2)}).")


if __name__ == '__main__':
    main()