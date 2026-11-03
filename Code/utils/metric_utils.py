from typing import Dict, Any, List
from typing import List, Optional, Tuple, Dict, Any
import os
import copy
import re
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom not in (0, None) else 0.0


def code_churn_rate(metadata: Dict[str, Any]) -> Tuple[List[float], List[float], List[float]]:
    adds = metadata.get('additions_per_year', [])
    dels = metadata.get('deletions_per_year', [])
    loc_start = metadata.get('total_loc_start_per_year', [])
    result = []
    result_month = []
    result_3week = []
    for i in range(len(metadata.get('years', []))):
        churn = _safe_div((adds[i] if i < len(adds) else 0) + (dels[i] if i < len(dels) else 0),
                          loc_start[i] if i < len(loc_start) else 0)
        if churn<0:
            print(1)
        result.append(round(churn * 100, 4))
        result_month.append(round(churn / 12 * 100, 4))
        result_3week.append(round(churn / (365/21) * 100, 4))
    return result, result_month, result_3week


def burstiness(metadata: Dict[str, Any]) -> List[float]:
    return metadata.get('burstiness_per_year', [])[:len(metadata.get('years', []))]


def commit_frequency(metadata: Dict[str, Any]) -> List[float]:
    return metadata.get('commit_density_per_year', [])[:len(metadata.get('years', []))]


def test_proportion(metadata: Dict[str, Any], test_info: Dict[str, Any]) -> List[float]:
    tests = test_info.get('test_lines_count_per_year', [])
    total = metadata.get('total_loc_snapshot_per_year', [])
    result = []
    for i in range(len(metadata.get('years', []))):
        test_count = tests[i] if i < len(tests) else 0
        total_count = total[i] if i < len(total) else 0
        production_code = total_count - test_count
        ratio = _safe_div(test_count, production_code)
        result.append(round(ratio * 100, 4))
    return result


def test_code_churn_rate(test_info: Dict[str, Any], code_churn_output: Tuple[List[float], List[float], List[float]]) -> Tuple[List[float], List[float], List[float], List[float]]:
    add = test_info.get('test_lines_added_per_year', [])
    dele = test_info.get('test_lines_deleted_per_year', [])
    start = test_info.get('test_lines_start_per_year', [])

    result = []
    result_month = []
    result_3week = []

    for i in range(len(test_info.get('years', []))):
        churn = _safe_div((add[i] if i < len(add) else 0) + (dele[i] if i < len(dele) else 0),
                          start[i] if i < len(start) else 0)
        result.append(round(churn * 100, 4))
        result_month.append(round(churn / 12 * 100, 4))
        result_3week.append(round(churn / (52/3) * 100, 4))

    # Calculate ratio of 3-week average test code churn rate to code churn rate
    code_churn_year, code_churn_month, code_churn_3week = code_churn_output
    ratio_3week = []
    for i in range(len(result_3week)):
        ratio = _safe_div(result_3week[i] if i < len(result_3week) else 0,
                         code_churn_3week[i] if i < len(code_churn_3week) else 0)
        ratio_3week.append(round(ratio, 4))

    return result, result_month, result_3week, ratio_3week


def community_activity(developer_info: Dict[str, Any]) -> List[float]:
    return developer_info.get('core_dev_turnover_per_year', [])[:len(developer_info.get('years', []))]


def truck_factor(developer_info: Dict[str, Any]) -> List[int]:
    return developer_info.get('truck_factor_per_year', [])[:len(developer_info.get('years', []))]


def newcomer_retention(developer_info: Dict[str, Any]) -> List[float]:
    result=[]
    retention=developer_info.get('newcomer_retention_next_year_per_year', [])[:len(developer_info.get('years', []))]
    for _value in retention:
        if _value==None:
            result.append(None)
        else:
            result.append(round(_value*100, 4))
    return result


def issue_response_efficiency(issue_info: Dict[str, Any]) -> List[float]:
    result=issue_info.get('avg_issue_resolution_days_per_year', [])[:len(issue_info.get('years', []))]
    if max(result)==0:
        result=[None for i in range(len(result))]
    return result


def backlog_management_index(issue_info: Dict[str, Any]) -> List[float]:
    closed = issue_info.get('closed_issues_per_year', [])
    opened = issue_info.get('issues_per_year', [])
    result = []
    for i in range(len(issue_info.get('years', []))):
        idx = _safe_div(closed[i] if i < len(closed) else 0, opened[i] if i < len(opened) else 0)
        result.append(round(idx * 100, 4))
    if max(opened)==0 and max(closed)!=0:
        result=[None for i in range(len(result))]
    return result


def cve_exposure_rate(issue_info: Dict[str, Any]) -> List[float]:
    counts = issue_info.get('cve_count_per_year', [])
    hist_avg = issue_info.get('historical_avg_cve_count', 0)
    result_div = []
    result_diff = []
    for i in range(len(issue_info.get('years', []))):
        n = counts[i]
        lambda_val = hist_avg
        # N - (λ + 3√λ)
        exp = n - (lambda_val + 3 * np.sqrt(lambda_val))
        if hist_avg==0: # The warining is only valid for has historical CVE.
            result_div.append(0)
            result_diff.append(-1)
        elif hist_avg>200 and n>200: # handle the mismatch in CVE API
            result_div.append(0)
            result_diff.append(-1)
        else:
            result_div.append(round(_safe_div(n,exp), 4))
            result_diff.append(float(round(exp, 4)))

    return result_div,result_diff


def cve_severity(issue_info: Dict[str, Any]) -> List[float]:
    return issue_info.get('avg_cvss_per_year', [])[:len(issue_info.get('years', []))]


def _filter_data_by_end_year(data_dict: Dict[str, Any], end_year: int) -> Dict[str, Any]:
    """
    Filter dictionary data that exceeds end_year

    Args:
        data_dict: Dictionary containing 'years' key and other list values
        end_year: Maximum allowed year

    Returns:
        Filtered dictionary
    """
    if not data_dict or 'years' not in data_dict:
        return data_dict

    years = data_dict.get('years', [])
    if not years:
        return data_dict

    # Find indices to keep (year <= end_year)
    valid_indices = [i for i, year in enumerate(years) if year <= end_year]

    if len(valid_indices) == len(years):
        # No data needs to be filtered
        return data_dict

    # Create new dict, filter all list values
    filtered_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, list) and len(value) == len(years):
            # Keep only elements at valid_indices
            filtered_dict[key] = [value[i] for i in valid_indices]
        else:
            # Non-list value or length mismatch, keep as-is
            filtered_dict[key] = value

    return filtered_dict


def our_score_card_metric(result: Dict[str, Any], end_year: int = 2025) -> Dict[str, Any]:
    """
    Compute all metrics per year using collected metadata.
    Args:
        result: output of our_score_card_scan (single repo)
        end_year: maximum year to include in metrics (default: 2025)
    Returns:
        dict containing per-year metric arrays and years
    """
    # Get raw data
    md = result.get('metadata', {})
    tests = result.get('test_info', {})
    dev = result.get('developer_info', {})
    issues = result.get('issue_vulnerability_info', {})

    # Filter out data exceeding end_year
    md = _filter_data_by_end_year(md, end_year)
    tests = _filter_data_by_end_year(tests, end_year)
    dev = _filter_data_by_end_year(dev, end_year)
    issues = _filter_data_by_end_year(issues, end_year)

    years = issues.get('years', [])

    code_churn_year, code_churn_month, code_churn_3week = code_churn_rate(md)
    test_code_churn_year, test_code_churn_month, test_code_churn_3week, test_code_churn_ratio_3week = test_code_churn_rate(tests, (code_churn_year, code_churn_month, code_churn_3week))
    cve_rate,cve_increase = cve_exposure_rate(issues)
    metrics = {
        'years': years,
        'code_churn_rate(%, year)': code_churn_year,
        'code_churn_rate(%, month)': code_churn_month,
        'code_churn_rate(%, 3week)': code_churn_3week,
        'burstiness(year)': burstiness(md),
        'commit_frequency(year)': commit_frequency(md),
        'test_proportion(%, year)': test_proportion(md, tests),
        'test_code_churn_rate(%, year)': test_code_churn_year,
        'test_code_churn_rate(%, month)': test_code_churn_month,
        'test_code_churn_rate(%, 3week)': test_code_churn_3week,
        'test_code_churn_ratio(3week)': test_code_churn_ratio_3week,
        'core_developer_turnover(year)': community_activity(dev),
        'truck_factor(year)': truck_factor(dev),
        'newcomer_retention(%, year)': newcomer_retention(dev),
        'issue_response_efficiency(day, year)': issue_response_efficiency(issues),
        'backlog_management_index(%, year)': backlog_management_index(issues),
        'cve_exposure_rate(year)': cve_rate,
        'cve_exposure_increase(year)': cve_increase,
        'cve_severity(year)': cve_severity(issues),
    }
    return metrics

#  ==================Predict==================


def arima_forecast(
    y: List[float],
    steps: Optional[int] = None,
    order: Optional[Tuple[int, int, int]] = None,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
) -> Dict[str, Any]:
    """
    ARIMA forecast for univariate time series (e.g., yearly/monthly data).

    Parameters:
    - y: List of numerical values in temporal order (cannot be all NaN/None)
    - steps: Number of future periods to forecast. If None, defaults to 20% of data length (floor, minimum 1)
    - order: (p, d, q). If None, uses AIC-based automatic order selection
    - max_p, max_d, max_q: Upper bounds for automatic order search (default 0..3 / 0..2 / 0..3)

    Returns:
    - dict: {
        "order": (p,d,q),
        "aic": float,
        "forecast": [..],
        "conf_int": [[lower, upper], ...],
        "model": fitted_model_object
      }
    """
    # Auto-calculate steps if not provided: 20% of data length, floor, minimum 1
    s = pd.Series(y, dtype="float64").dropna()
    if steps is None:
        steps = max(1, int(len(s) * 0.2))

    if steps <= 0:
        raise ValueError("steps must be a positive integer.")

    def fit_aic(p: int, d: int, q: int):
        try:
            model = ARIMA(
                s,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit()
            return res.aic, res
        except Exception:
            return np.inf, None

    best_res = None
    best_aic = np.inf
    best_order = None

    if order is not None:
        p, d, q = order
        best_aic, best_res = fit_aic(p, d, q)
        best_order = order
        if best_res is None:
            raise ValueError(f"Failed to fit with given order={order}. Try a different order or check data.")
    else:
        for d in range(0, max_d + 1):
            for p in range(0, max_p + 1):
                for q in range(0, max_q + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    aic, res = fit_aic(p, d, q)
                    if aic < best_aic and res is not None:
                        best_aic, best_res = aic, res
                        best_order = (p, d, q)

        if best_res is None:
            raise RuntimeError("Automatic order selection failed: all candidate ARIMA models failed to fit.")

    pred = best_res.get_forecast(steps=steps)
    forecast = pred.predicted_mean.to_list()
    conf_int = pred.conf_int(alpha=0.05).to_numpy().tolist()

    return {
        "order": best_order,
        "aic": float(best_aic),
        "forecast": forecast,
        "conf_int": conf_int,
        "model": best_res,
    }

def process_alarm(alarm_dict_raw: Dict[str, list]) -> Dict[str, list]:
    # Optional, solve False Positives
    if list(alarm_dict_raw.keys())==['truck_factor(year)']:
        alarm_dict={}
    else:
        alarm_dict=alarm_dict_raw
    return alarm_dict

def visualize_arima_forecast(
    historical_data: List[float],
    forecast_result: Dict[str, Any],
    metric_name: str,
    years: Optional[List[int]] = None,
    save_dir: str = "exp_result",
    filename: Optional[str] = None,
) -> None:
    """
    Visualize ARIMA forecast results as a continuous line with historical data (solid line)
    and forecasted data (dashed line) connected together.

    Parameters:
    - historical_data: List of historical values
    - forecast_result: Output from arima_forecast function
    - metric_name: Name of the metric being visualized (for plot title)
    - years: List of years corresponding to historical data. If None, uses sequential indices
    - save_dir: Directory to save the plot (default: "exp_result")
    - filename: Output filename. If None, auto-generates from metric_name
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Extract forecast data
    forecast = forecast_result.get("forecast", [])

    # Remove None values from historical data for plotting
    historical_clean = [val if val is not None else np.nan for val in historical_data]

    # Generate x-axis values
    n_hist = len(historical_data)
    n_forecast = len(forecast)

    if years is not None and len(years) == n_hist:
        # Use provided years and extend for forecast
        # Note: years might be in descending order, find the actual last (maximum) year
        x_hist = years
        last_year = max(years)  # Use max to get the latest year regardless of order
        # Generate forecast years starting from last_year + 1
        x_forecast = list(range(last_year + 1, last_year + 1 + n_forecast))
    else:
        # Use sequential indices
        x_hist = list(range(1, n_hist + 1))
        x_forecast = list(range(n_hist + 1, n_hist + 1 + n_forecast))

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot historical part (blue solid line with circle markers)
    plt.plot(x_hist, historical_clean, marker='o', linestyle='-', linewidth=2,
             color='blue', markersize=6, label='Historical Data')

    # Plot connection from last historical point to first forecast point (dashed line)
    # Find the index of the maximum year in x_hist to get the correct last historical point
    if years is not None:
        last_hist_idx = x_hist.index(last_year)
        last_hist_value = historical_clean[last_hist_idx]
    else:
        last_hist_idx = -1
        last_hist_value = historical_clean[-1]

    connection_x = [last_year, x_forecast[0]]
    connection_y = [last_hist_value, forecast[0]]
    plt.plot(connection_x, connection_y, linestyle='--', linewidth=2, color='red', alpha=0.7)

    # Plot forecast part (red dashed line with square markers)
    plt.plot(x_forecast, forecast, marker='s', linestyle='--', linewidth=2,
             color='red', markersize=6, label='Forecast')

    # Customize plot
    plt.xlabel('Year' if years is not None else 'Time Period', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'ARIMA Forecast: {metric_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Generate filename if not provided
    if filename is None:
        # Clean metric name for filename (remove special characters)
        clean_name = "".join(c if c.isalnum() else "_" for c in metric_name)
        filename = f"arima_forecast_{clean_name}.png"

    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {save_path}")

