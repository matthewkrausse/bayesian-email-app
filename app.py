import streamlit as st
import math
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import pandas as pd # Using pandas for better display of inputs

# --- Updated Calculation Function ---
def calculate_sample_size(baseline_rate, mde, alpha, power, num_groups=2, alternative='two-sided'):
    """
    Calculates the required sample size per group for an A/B/.../N test
    comparing multiple proportions against a control, using Bonferroni correction.

    Args:
        baseline_rate (float): The expected proportion for the control group (p1).
        mde (float): The Minimum Detectable Effect vs control you want to detect.
        alpha (float): The desired overall significance level (FWER).
        power (float): The desired statistical power for each comparison against control.
        num_groups (int): The total number of groups (control + variants). Must be >= 2.
        alternative (str): The alternative hypothesis ('two-sided', 'larger', 'smaller').

    Returns:
        int: The required sample size per group, rounded up.
        float: The target rate for the variation group (p2).
        float: The adjusted alpha used per comparison.

    Raises:
        ValueError: If input parameters are invalid.
    """
    # Input Validation
    if not 0 < baseline_rate < 1:
        raise ValueError("Baseline rate must be strictly between 0 and 1.")
    if mde <= 0:
        raise ValueError("Minimum Detectable Effect (MDE) must be positive.")
    if not 0 < alpha < 1:
        raise ValueError("Overall Alpha (significance level) must be strictly between 0 and 1.")
    if not 0 < power < 1:
        raise ValueError("Power must be strictly between 0 and 1.")
    if not isinstance(num_groups, int) or num_groups < 2:
        raise ValueError("Number of groups must be an integer of at least 2.")

    p1 = baseline_rate
    p2 = baseline_rate + mde

    # Check if calculated target rate is valid
    if not 0 < p2 < 1:
        raise ValueError(
            f"Calculated target rate (baseline + MDE = {p2:.4f}) is outside the valid range (0, 1). "
            "Adjust baseline rate or MDE."
        )

    if alternative not in ['two-sided', 'larger', 'smaller']:
        raise ValueError("Alternative must be 'two-sided', 'larger', or 'smaller'.")

    # --- Bonferroni Correction ---
    # Assuming k-1 comparisons against the control for k groups
    num_comparisons = num_groups - 1
    if num_comparisons < 1: # Should be caught by num_groups check, but belt-and-suspenders
        adjusted_alpha = alpha
    else:
        adjusted_alpha = alpha / num_comparisons
        if adjusted_alpha >= alpha: # Sanity check
            adjusted_alpha = alpha # Should only happen if num_comparisons is 1

    if adjusted_alpha <= 0:
         raise ValueError(f"Adjusted alpha ({adjusted_alpha:.2e}) is too low. Check overall alpha and number of groups.")

    # Calculate effect size using Cohen's h
    try:
        effect_size = proportion_effectsize(p1, p2)
    except ValueError as e:
        raise ValueError(f"Could not calculate effect size. Check inputs. Original error: {e}")

    # Initialize the power analysis class
    analysis = NormalIndPower()

    # Solve for the number of observations (sample size) per group using ADJUSTED alpha
    try:
        sample_size_float = analysis.solve_power(
            effect_size=abs(effect_size),
            alpha=adjusted_alpha, # Use adjusted alpha here
            power=power,
            ratio=1.0,
            alternative=alternative
        )
    except Exception as e:
         raise RuntimeError(f"Statsmodels power calculation failed. Check inputs. Original error: {e}")

    # Return sample size rounded up, target rate, and adjusted alpha
    sample_size_int = math.ceil(sample_size_float)
    return sample_size_int, p2, adjusted_alpha

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")

st.title("📊 Multi-Group A/B Test Sample Size Calculator")
st.markdown("""
This app calculates the minimum sample size **per group** required for an A/B/.../N test
comparing multiple variations against a control group. It uses the **Bonferroni correction**
to control the overall false positive rate (Family-Wise Error Rate).
""")

st.sidebar.header("Input Parameters")

# --- Input fields ---
num_groups = st.sidebar.number_input(
    label="Number of Groups (Total)",
    min_value=2,
    value=2,
    step=1,
    help="Total number of versions being tested, including the control (e.g., 3 for A/B/C test)."
)

baseline_rate = st.sidebar.number_input(
    label="Baseline Conversion Rate (p1)",
    min_value=0.0001, max_value=0.9999, value=0.20, step=0.01, format="%.4f",
    help="The expected conversion rate of your control group."
)

mde = st.sidebar.number_input(
    label="Minimum Detectable Effect (MDE)",
    min_value=0.0001, value=0.02, step=0.001, format="%.4f",
    help="The smallest absolute difference *vs control* you want to detect (e.g., 0.02 for a 2% increase)."
)

alpha = st.sidebar.slider(
    label="Overall Significance Level (α)",
    min_value=0.01, max_value=0.25, value=0.05, step=0.01, format="%.2f",
    help="Desired overall probability of making at least one Type I error (false positive) across all comparisons. Typically 0.05."
)

power = st.sidebar.slider(
    label="Statistical Power (1-β)",
    min_value=0.50, max_value=0.99, value=0.80, step=0.01, format="%.2f",
    help="Desired probability of detecting the MDE *for a specific comparison* if it truly exists. Typically 0.80."
)

alternative = st.sidebar.selectbox(
    label="Alternative Hypothesis",
    options=['two-sided', 'larger', 'smaller'], index=0,
    help="'two-sided': Test p_variant != p_control. 'larger': Test p_variant > p_control."
)

# --- Calculate Button ---
calculate_button = st.sidebar.button("Calculate Sample Size", type="primary")

st.divider()

# --- Display Results ---
st.header("Results")

if calculate_button:
    try:
        # Perform calculation
        sample_size_per_group, target_rate, adjusted_alpha = calculate_sample_size(
            baseline_rate, mde, alpha, power, num_groups, alternative
        )
        total_sample_size = sample_size_per_group * num_groups

        st.success("Calculation Successful!")

        st.markdown("#### Required Sample Sizes:")
        col1, col2 = st.columns(2)
        with col1:
            # Use comma for thousands separator
            st.metric(label=f"Sample Size Per Group ({num_groups} Groups)", value=f"{sample_size_per_group:,}")
        with col2:
            st.metric(label="Total Sample Size (All Groups)", value=f"{total_sample_size:,}")

        st.markdown("#### Calculation Inputs & Parameters:")
        # Use a DataFrame for slightly nicer display of parameters
        param_data = {
            "Parameter": ["Baseline Rate (p1)", "Minimum Detectable Effect (MDE)", "Target Rate (p2)",
                          "Overall Significance Level (Alpha)", "Adjusted Alpha per Comparison",
                          "Statistical Power (1-Beta)", "Number of Groups", "Alternative Hypothesis"],
            "Value": [f"{baseline_rate:.4f}", f"{mde:.4f}", f"{target_rate:.4f}",
                      f"{alpha:.3f}", f"{adjusted_alpha:.5f}", f"{power:.2f}",
                      num_groups, alternative]
        }
        st.dataframe(pd.DataFrame(param_data), hide_index=True)


        st.info(f"""
        **Note on Correction:** The calculation uses the Bonferroni correction by adjusting the significance level
        for each comparison against the control to **{adjusted_alpha:.5f}** (Overall Alpha / {num_groups-1} comparisons).
        This helps control the overall chance of a false positive but can be conservative (potentially requiring more samples than other methods).
        """, icon="ℹ️")


    except (ValueError, RuntimeError) as e:
        st.error(f"Error during calculation: {e}", icon="🚨")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}", icon="🔥")
else:
    st.info("Adjust the parameters in the sidebar and click 'Calculate Sample Size'.")

with st.expander("ℹ️ Learn More About the Methodology"):
    st.markdown("""
    This calculator helps you determine the necessary sample size for your A/B/.../N test based on
    statistical power analysis. Here's a breakdown of the key concepts:

    **1. Goal of A/B/.../N Testing:**
    The aim is to compare different versions (variants) of something against a baseline (control)
    to see which performs better based on a specific metric (e.g., open rate, conversion rate).
    We start by assuming the **Null Hypothesis (H₀)** is true: that there is *no difference* in the metric
    between the control and the variant (`p_variant = p_control`). The test seeks evidence to reject this null hypothesis.

    **2. Key Inputs:**
    * **Baseline Rate (p1):** Your starting point – the current performance of the control group.
    * **Minimum Detectable Effect (MDE):** The smallest *difference* you want to detect.
    * **Statistical Power (1-β):** Probability of detecting the MDE if it's real (sensitivity).
    * **Overall Significance Level (α):** Acceptable probability of a *false positive* (Type I error).

    **3. Handling Multiple Groups (Bonferroni Correction):**
    When `Number of Groups > 2`, the chance of a false positive *somewhere* increases (Family-Wise Error Rate - FWER).
    This calculator uses **Bonferroni correction** (dividing overall α by the number of comparisons against control)
    to maintain the desired overall FWER. This is simple but can be conservative.

    **4. Alternative Hypotheses (H₁):**
    This setting defines what kind of difference you are trying to detect, changing how the statistical test evaluates the evidence against the Null Hypothesis (H₀: `p_variant = p_control`).
    * **`two-sided` (Default):**
        * *Hypothesis:* Is the variant rate simply *different* from the control rate? (H₁: `p_variant ≠ p_control`)
        * *Use Case:* Choose this when you don't know beforehand if the change will cause an increase or decrease, or if you care about detecting a difference in *either* direction.
        * *Impact:* This is the most common choice but requires the largest sample size compared to one-sided tests (for the same MDE, Power, Alpha).
    * **`larger`:**
        * *Hypothesis:* Is the variant rate *greater than* the control rate? (H₁: `p_variant > p_control`)
        * *Use Case:* Choose this when you are specifically testing for an *improvement* and would not act on, or don't care about detecting, a decrease. Example: Testing if a new feature *increases* sign-ups.
        * *Impact:* Requires slightly fewer samples than `two-sided` because it focuses the test's power on one direction.
    * **`smaller`:**
        * *Hypothesis:* Is the variant rate *less than* the control rate? (H₁: `p_variant < p_control`)
        * *Use Case:* Choose this when you are specifically testing for a *decrease*. Example: Testing if a change *reduces* unsubscribe rates or error occurrences.
        * *Impact:* Similar to `larger`, requires slightly fewer samples than `two-sided`.

    **5. The Calculation:**
    The app uses the `statsmodels` Python library, specifically its power analysis functions for comparing two proportions
    (based on normal approximations / Z-tests). It solves for the sample size (`nobs1`) needed to achieve the desired
    power, given the baseline rate, effect size (derived from MDE), the (adjusted) significance level, and the chosen alternative hypothesis.

    **Further Reading:**
    * **Hypothesis Testing:** [Wikipedia](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
    * **Statistical Power:** [Wikipedia](https://en.wikipedia.org/wiki/Statistical_power) | [NIST Handbook](https://www.itl.nist.gov/div898/handbook/prc/section2/prc22.htm)
    * **A/B Testing Statistics:** [Evan Miller's Awesome A/B Tools](https://www.evanmiller.org/ab-testing/) | [Wikipedia](https://en.wikipedia.org/wiki/A/B_testing#Statistical_significance)
    * **Significance Level (Alpha) & P-values:** [Wikipedia](https://en.wikipedia.org/wiki/Statistical_significance)
    * **Family-Wise Error Rate (FWER):** [Wikipedia](https://en.wikipedia.org/wiki/Family-wise_error_rate)
    * **Bonferroni Correction:** [Wikipedia](https://en.wikipedia.org/wiki/Bonferroni_correction) | [Statistics How To](https://www.statisticshowto.com/bonferroni-correction/)
    * **Statsmodels Power Analysis:** [Statsmodels Documentation](https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations)
    """)

st.divider()
st.markdown("Built with [Streamlit](https://streamlit.io) | Uses [statsmodels](https://www.statsmodels.org/)")