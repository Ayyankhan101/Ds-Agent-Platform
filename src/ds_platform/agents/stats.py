import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional


class StatsAgent:
    """
    Agent for Hypothesis Testing and Statistical Inference.
    """

    def run_test(self, df: pd.DataFrame, test_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        result = {"test_type": test_type}

        try:
            alpha = params.get("alpha", 0.05)

            if test_type == "t-test":
                col = params.get("column_a") or params.get("column")
                popmean = params.get("popmean", 0)
                if col and col in df.columns:
                    data = df[col].dropna()
                    t_stat, p_val = stats.ttest_1samp(data, popmean)
                    result.update(
                        {
                            "statistic": float(t_stat),
                            "p_value": float(p_val),
                            "degrees_of_freedom": len(data) - 1,
                        }
                    )
                else:
                    result["error"] = "Column not found"

            elif test_type == "chi-square":
                col1 = params.get("column_a") or params.get("column1")
                col2 = params.get("column_b") or params.get("column2")
                if col1 and col2:
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                    result.update(
                        {
                            "statistic": float(chi2),
                            "p_value": float(p_val),
                            "dof": dof,
                        }
                    )
                else:
                    result["error"] = "Chi-square requires two columns"

            elif test_type == "anova":
                col_a = params.get("column_a")
                col_b = params.get("column_b")
                if col_a and col_b:
                    groups = df[col_b].unique()
                    group_data = [
                        df[df[col_b] == g][col_a].dropna()
                        for g in groups
                        if len(df[df[col_b] == g]) > 0
                    ]
                    if len(group_data) >= 2:
                        f_stat, p_val = stats.f_oneway(*group_data)
                        result.update(
                            {
                                "statistic": float(f_stat),
                                "p_value": float(p_val),
                                "groups": len(groups),
                            }
                        )
                    else:
                        result["error"] = "ANOVA requires at least 2 groups"
                else:
                    result["error"] = "ANOVA requires column selection"

            elif test_type == "mann-whitney":
                col_a = params.get("column_a")
                col_b = params.get("column_b")
                if col_a and col_b:
                    data1 = df[col_a].dropna()
                    data2 = df[col_b].dropna()
                    stat, p_val = stats.mannwhitneyu(data1, data2)
                    result.update(
                        {
                            "statistic": float(stat),
                            "p_value": float(p_val),
                            "n1": len(data1),
                            "n2": len(data2),
                        }
                    )
                else:
                    result["error"] = "Mann-Whitney requires two columns"

            else:
                result["error"] = f"Unknown test type: {test_type}"

            # Add interpretation
            if "p_value" in result:
                result["interpretation"] = (
                    "Reject H0" if result["p_value"] < alpha else "Fail to reject H0"
                )
                result["alpha"] = alpha

        except Exception as e:
            result["error"] = str(e)

        return result
