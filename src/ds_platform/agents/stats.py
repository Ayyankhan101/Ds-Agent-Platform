import pandas as pd
from scipy import stats
from typing import Dict, Any

class StatsAgent:
    """
    Agent for Hypothesis Testing and Statistical Inference.
    """
    def run_test(self, df: pd.DataFrame, test_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        result = {"test_type": test_type}
        
        try:
            if test_type == "t-test":
                col = params.get("column")
                popmean = params.get("popmean", 0)
                t_stat, p_val = stats.ttest_1samp(df[col].dropna(), popmean)
                result.update({"statistic": t_stat, "p_value": p_val})
            
            elif test_type == "chi-square":
                col1 = params.get("column1")
                col2 = params.get("column2")
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                result.update({"statistic": chi2, "p_value": p_val, "dof": dof})

            result["interpretation"] = "Reject H0" if result["p_value"] < 0.05 else "Fail to reject H0"
        except Exception as e:
            result["error"] = str(e)

        return result
