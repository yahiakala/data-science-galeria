"""Functions for tax rates."""

import numpy as np
import pandas as pd

ei_table_url = 'https://www.canada.ca/en/services/benefits/ei/ei-regular-benefit/benefit-amount.html'  # noqa


def federal_rate(annual_gross_income, brackets, rates):
    """Return the marginal federal tax rate."""
    # Upper limit on brackets is arbitrary.
    for i, j in zip(brackets, rates):
        if annual_gross_income <= i:
            marginal_rate = j
            break
    return marginal_rate


def average_rate(annual_gross_income, brackets, rates):
    """Return the average federal tax rate."""
    # Get the index of the upper bracket
    for i in brackets:
        if annual_gross_income <= i:
            upper_bracket_index = brackets.index(i) + 1
            break

    tax_brackets_new = brackets[:upper_bracket_index].copy()

    # Set the upper bracket to the annual gross income
    tax_brackets_new[-1] = annual_gross_income

    # Get diffs and prepend first bracket
    bracket_diffs = [brackets[0]] + list(np.diff(tax_brackets_new))

    # Weighted average using the two lists
    total_tax = np.dot(bracket_diffs, rates[:upper_bracket_index])
    return total_tax / annual_gross_income


def premium_deduction(taxable_income, premium=0.0158, maximum_income=56300):
    """Return the Annual EI or CPP Premium."""
    max_income = min(taxable_income, maximum_income)
    return max_income * premium / max_income


# Less General Functions
def calc_takehome(taxable_income, federal, provincial, cpp, ei):
    """Calculate take-home taxable income."""
    return None


def ei_benefits(taxable_income):
    """Calculate EI benefits. No consid for variable pay on a weekly basis."""
    # Total insurable earnings
    insurable_earnings = 56300 / 52 * 0.55  # noqa Assume constant
    # Then tax them both federally and provincially, but no CPP/EI deductions.
    # Then multiply by max number of weeks depending on unemployment rate.


def scrape_table_EI(webpage_url=ei_table_url):
    """Pull the max weeks table into a DataFrame."""
    tables = pd.read_html(webpage_url)[0]  # noqa
    # Get upper limits for both column names and num hours names
