import os
import pickle
from tqdm import trange
from pathlib import Path
import re
import concurrent.futures as _futures
import json
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# from matplotlib import font_manager as fm
# font_path = "TIMES.TTF"
# import matplotlib.font_manager as fm
# fm.fontManager.addfont(font_path)
# prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = prop.get_name()
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau

from typing import List, Tuple, Dict
from itertools import combinations
import matplotlib.colors as mcolors
from math import sqrt, asin
import math
from typing import List, Tuple, Dict, Any
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch
import csv
import pickle
import json
from datetime import date
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


def filter_repos_by_ossf_threshold(
    ossf_by_year: Dict[str, Dict[int, float]],
    output_dir: str = 'exp_result/0_rq'
) -> None:
    """
    Scan ossf_by_year, filter repos by threshold and save to CSV files.

    For each repo, check if scores for all years are greater than the specified threshold.
    Skip repos where any year has a None value.

    Args:
        ossf_by_year: {repo_name: {year: score, ...}, ...}
        output_dir: Directory for CSV output files
    """
    thresholds = [9, 7, 5]
    results = {t: [] for t in thresholds}

    for repo_name, year_scores in ossf_by_year.items():
        scores = list(year_scores.values())
        # Skip if any value is None
        if any(s is None for s in scores):
            continue
        # Check each threshold
        for t in thresholds:
            if all(s > t for s in scores):
                results[t].append(repo_name)

    # Save to CSV files
    os.makedirs(output_dir, exist_ok=True)
    for t in thresholds:
        csv_path = os.path.join(output_dir, f'repos_score-{t}.csv')
        if os.path.exists(csv_path):
            print(f"{csv_path} already exists, skipping")
            continue
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for repo_name in sorted(results[t]):
                writer.writerow([repo_name])
        print(f"Saved {len(results[t])} repos to {csv_path}")


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (lower, upper) as proportions in [0, 1].
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    # z for two-sided alpha (approx via inverse error function is messy without scipy)
    # We'll use a small lookup for common alphas; fallback to normal approx using math.erfcinv if available (not in stdlib).
    # For typical 95% CI: alpha=0.05 => z≈1.959964
    z_lookup = {
        0.10: 1.6448536269514722,  # 90%
        0.05: 1.959963984540054,   # 95%
        0.01: 2.5758293035489004,  # 99%
    }
    z = z_lookup.get(alpha)
    if z is None:
        # Fallback: use 95% if alpha is unusual (keeps function dependency-free)
        z = z_lookup[0.05]

    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    return lower, upper


def rq1_plot_score_alarm_buckets(
    data,
    bin_width: float = 1.0,
    score_min: float = 0.0,
    score_max: float = 10.0,
    alpha: float = 0.05,
    figsize=(6.5, 4),
    ax=None,
    fig_path: str = '',
    tick_fontsize: int = 18,
    label_fontsize: int = 18,
    annotate_fontsize: int = 18
):
    """
    Visualize alarm rate (%) by score buckets.

    data can be in two formats:
      1. repo_dict: {repo_name: {"ossf_score": float, "our_triggered": list}}
      2. bucket_stats: [{'bucket': '0-1', 'n_total': int, 'k_alarm': int, 'alarm_rate': float}, ...]
    """
    # Determine input type: if list then bucket_stats, otherwise repo_dict
    if isinstance(data, list):
        # Use bucket_stats directly for plotting
        labels = [b['bucket'] for b in data]
        n_per = np.array([b['n_total'] for b in data])
        k_per = np.array([b['k_alarm'] for b in data])
        y = np.array([b['alarm_rate'] for b in data])
        centers = np.arange(len(data)) + 0.5
        bucket_stats = data
        scores_np, alarms_np = None, None
    else:
        # Original logic: extract data from repo_dict
        repo_dict = data
        scores: List[float] = []
        alarms: List[int] = []
        repo_names: List[str] = []
        triggered_info: List[List] = []
        for repo, v in repo_dict.items():
            if not isinstance(v, dict):
                continue
            s = v["ossf_score"]
            a = v["our_triggered"]
            if s is None:
                continue
            alarm_flag = 1 if (isinstance(a, list) and len(a) > 0) else 0
            scores.append(s)
            alarms.append(alarm_flag)
            repo_names.append(repo)
            triggered_info.append(a if isinstance(a, list) else [])

        if len(scores) == 0:
            raise ValueError("No valid (score, alarm) pairs found in repo_dict.")

        scores_np = np.asarray(scores, dtype=float)
        alarms_np = np.asarray(alarms, dtype=int)

        edges = np.arange(score_min, score_max + bin_width, bin_width, dtype=float)
        if edges[-1] < score_max:
            edges = np.append(edges, score_max)
        nbins = len(edges) - 1

        bin_idx = np.digitize(scores_np, edges, right=False) - 1
        bin_idx = np.where(scores_np == score_max, nbins - 1, bin_idx)

        in_range = (scores_np >= score_min) & (scores_np <= score_max) & (bin_idx >= 0) & (bin_idx < nbins)
        bin_idx_filtered = bin_idx[in_range]
        alarms_in = alarms_np[in_range]
        repo_names_filtered = [repo_names[i] for i, flag in enumerate(in_range) if flag]
        triggered_filtered = [triggered_info[i] for i, flag in enumerate(in_range) if flag]

        n_per = np.zeros(nbins, dtype=int)
        k_per = np.zeros(nbins, dtype=int)
        bucket_alarmed_repos: List[Dict[str, List]] = [{} for _ in range(nbins)]
        for bin_i, alarm_flag, repo, triggered in zip(bin_idx_filtered, alarms_in, repo_names_filtered, triggered_filtered):
            n_per[bin_i] += 1
            k_per[bin_i] += int(alarm_flag)
            if alarm_flag == 1:
                bucket_alarmed_repos[bin_i][repo] = triggered

        lefts = edges[:-1]
        rights = edges[1:]
        centers = lefts + (rights - lefts) / 2.0
        labels = []
        for l, r in zip(lefts, rights):
            if float(l).is_integer() and float(r).is_integer():
                labels.append(f"{int(l)}-{int(r)}")
            else:
                labels.append(f"{l:g}-{r:g}")

        y = np.array([k_per[i] / n_per[i] * 100 if n_per[i] > 0 else 0 for i in range(nbins)])

        bucket_stats = []
        for i in range(nbins):
            bucket_stats.append({
                'bucket': labels[i],
                'n_total': int(n_per[i]),
                'k_alarm': int(k_per[i]),
                'alarm_rate': float(y[i]),
                'alarmed_repos': bucket_alarmed_repos[i]
            })

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.bar(centers, y, width=0.85, alpha=0.7, edgecolor='black')
    ax.set_xlabel("OpenSSF Score Range", fontsize=label_fontsize)
    ax.set_ylabel("Alert Rate (%)", fontsize=label_fontsize)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylim(0, max(y) * 1.1 if max(y) > 0 else 100)

    for x, k, n, rate_val in zip(centers, k_per, n_per, y):
        if n == 0:
            continue
        ax.annotate(f"{k}", (x, rate_val),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=annotate_fontsize)

    plt.subplots_adjust(top=0.96, bottom=0.22, right=0.96, left=0.12)
    fig.savefig(fig_path)

    return scores_np, alarms_np, bucket_stats


def rq1_plot_alarm_rate_lines(
    bucket_dict: Dict[int, List[Dict]],
    combined_bucket: List[Dict],
    figsize=(6.5, 4),
    fig_path: str = '',
    tick_fontsize: int = 18,
    label_fontsize: int = 18,
    year_colors: Dict[int, str] = None,
    year_linewidth: float = 1.5,
    combined_linewidth: float = 3.0,
):
    """
    Plot multi-year alarm rate line chart.

    Args:
        bucket_dict: {year: [{'bucket': '0-1', 'alarm_rate': float, ...}, ...]}
        combined_bucket: [{'bucket': '0-1', 'alarm_rate': float, ...}, ...]
        figsize: Figure size
        fig_path: Save path
        tick_fontsize: Tick font size
        label_fontsize: Label font size
        year_colors: Color for each year, auto-assigned if None
        year_linewidth: Line width for each year
        combined_linewidth: Line width for combined line
    """
    if year_colors is None:
        year_colors = {2023: '#7fc97f', 2024: '#beaed4', 2025: '#fdc086'}

    fig, ax = plt.subplots(figsize=figsize)

    # Get x-axis labels and positions
    labels = [b['bucket'] for b in combined_bucket]
    x = np.arange(len(labels))

    # Plot lines for each year
    for year in sorted(bucket_dict.keys()):
        y = np.array([b['alarm_rate'] for b in bucket_dict[year]])
        ax.plot(x, y, marker='o', linewidth=year_linewidth,
                color=year_colors.get(year, None), label=str(year), alpha=0.8)

    # Plot combined line (thicker red line)
    y_combined = np.array([b['alarm_rate'] for b in combined_bucket])
    ax.plot(x, y_combined, marker='o', linewidth=combined_linewidth,
            color='red', label='Total', zorder=10)

    ax.set_xlabel("OpenSSF Score Range", fontsize=label_fontsize)
    ax.set_ylabel("Alert Rate (%)", fontsize=label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend(fontsize=tick_fontsize, ncol=2)

    # y_max = max(max(b['alarm_rate'] for b in combined_bucket),
    #             max(max(b['alarm_rate'] for b in bucket_dict[y]) for y in bucket_dict))
    # ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 100)

    plt.subplots_adjust(top=0.96, bottom=0.22, right=0.96, left=0.15)

    if fig_path:
        fig.savefig(fig_path)
        print(f"Saved to {fig_path}")

    return fig, ax


def rq1_logistic_alarm_hypothesis_test(
    repo_dict: Dict[str, Dict[str, Any]],
    alpha: float = 0.05,
    score_min: float = 0.0,
    score_max: float = 10.0,
    robust: bool = True,
    score_points: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Logistic regression: hypothesis test and interpretive output for alarm ~ score

    H0: beta_score = 0   (score is unrelated to alarm)
    H1: beta_score != 0  (score is related to alarm)

    Args:
      repo_dict: {repo: {"score": float, "alarm": list}}
                alarm empty => 0, non-empty => 1
      alpha: significance level (default 0.05 => 95% CI)
      score_min/score_max: score range filter
      robust: True uses robust standard errors (HC1), more stable for mild heteroscedasticity
      score_points: list of score points for predicted probability output, default [2, 5, 8]

    Returns:
      dict containing direction, OR effect size, CI, p-value, LRT, etc.
    """
    try:
        import statsmodels.api as sm
        from scipy.stats import chi2
    except Exception as e:
        raise ImportError(
            "statsmodels and scipy are required for logistic regression hypothesis test output. "
            "Install with `pip install statsmodels scipy`. Original error: %r" % (e,)
        )

    if score_points is None:
        score_points = [2.0, 5.0, 8.0]

    # 1) Extract data
    scores = []
    y = []
    for _, v in repo_dict.items():
        s = v["ossf_score"]
        a = v["our_triggered"]
        if s is None:
            continue
        try:
            s = float(s)
        except (TypeError, ValueError):
            continue
        if not (score_min <= s <= score_max):
            continue
        alarm_flag = 1 if (isinstance(a, list) and len(a) > 0) else 0
        scores.append(s)
        y.append(alarm_flag)

    X = np.asarray(scores, dtype=float)
    Y = np.asarray(y, dtype=int)

    # 2) Model with intercept
    X_design = sm.add_constant(X, has_constant="add")  # [1, score]
    model = sm.Logit(Y, X_design)

    # Fit
    res = model.fit(disp=False, cov_type="HC1")

    # Robust standard errors (optional)
    if robust:
        res_rob = model.fit(disp=False, cov_type="HC1")
        used = res
        cov_note = "HC1 robust SE"
    else:
        used = res
        cov_note = "standard SE"

    # 3) Extract coefficients and test (Wald z-test)
    beta0, beta1 = float(used.params[0]), float(used.params[1])
    se1 = float(used.bse[1])
    z1 = beta1 / se1 if se1 > 0 else float("nan")
    p1 = float(used.pvalues[1])

    # Direction (interpretation: how alarm probability changes as score increases)
    if beta1 > 0:
        direction = "Higher score, higher alarm probability (positive correlation)"
    elif beta1 < 0:
        direction = "Higher score, lower alarm probability (negative correlation; low scores more likely to alarm)"
    else:
        direction = "No direction observed (coefficient is 0)"

    # 4) Effect size: OR = exp(beta1)
    OR_up_1 = math.exp(beta1)           # odds multiplier for score +1
    OR_down_1 = math.exp(-beta1)        # odds multiplier for score -1

    # 5) Confidence interval (for beta1, then exponentiate to get OR CI)
    # statsmodels conf_int defaults to coefficient CI
    ci_beta = used.conf_int(alpha=alpha)
    beta1_low, beta1_high = float(ci_beta[1, 0]), float(ci_beta[1, 1])
    OR_low, OR_high = math.exp(beta1_low), math.exp(beta1_high)

    # 6) Overall model significance: LRT (compared to intercept-only model)
    # LR = 2*(LL_full - LL_null) ~ chi2(df=1)
    ll_full = float(res.llf)
    # null model: only intercept (constant term only, no additional columns)
    X0 = np.ones((len(X), 1))
    res0 = sm.Logit(Y, X0).fit(disp=False)
    ll_null = float(res0.llf)
    LR = 2.0 * (ll_full - ll_null)
    p_LR = float(chi2.sf(LR, df=1))

    # 7) Predicted probability (for typical score points)
    def sigmoid(t: float) -> float:
        return 1.0 / (1.0 + math.exp(-t))

    pred_probs = {}
    for sp in score_points:
        sp = float(sp)
        pred_probs[sp] = sigmoid(beta0 + beta1 * sp)

    # 8) Provide a more readable comparison: probability difference between high and low score points
    if len(score_points) >= 2:
        s_lo = min(score_points)
        s_hi = max(score_points)
        delta = pred_probs[s_lo] - pred_probs[s_hi]
    else:
        s_lo = s_hi = score_points[0]
        delta = 0.0

    # 9) Output summary
    # Simple confidence-like expression: 1 - p (not strict statistical confidence, but commonly used intuitive expression)
    confidence_like = max(0.0, min(1.0, 1.0 - p1))

    out = {
        "n": int(len(Y)),
        "alarm_rate_overall": float(Y.mean()),
        "model": "Logit(alarm ~ score)",
        "covariance": cov_note,

        "direction": direction,

        # Core hypothesis test (Wald)
        "beta_score": beta1,
        "se_score": se1,
        "z_score": z1,
        "p_value_score": p1,
        "significant_at_alpha": bool(p1 < alpha),
        "alpha": alpha,

        # Effect size (odds ratio)
        "odds_ratio_per_score_plus_1": OR_up_1,
        "odds_ratio_per_score_minus_1": OR_down_1,
        "odds_ratio_CI": (OR_low, OR_high),

        # Overall model LRT
        "lr_stat_vs_intercept_only": LR,
        "lr_p_value": p_LR,

        # Predicted probability
        "predicted_alarm_prob_by_score": pred_probs,
        "prob_diff_low_minus_high": {
            "score_low": float(s_lo),
            "score_high": float(s_hi),
            "p_low": float(pred_probs[s_lo]),
            "p_high": float(pred_probs[s_hi]),
            "p_low_minus_p_high": float(delta),
        },

        # Intuitive confidence-like expression (for reporting convenience)
        "confidence_like_1_minus_p": confidence_like,
    }
    return out


def save_dict_as_json(d: dict, path: str = 'tmp.json'):
    """Save dictionary as formatted JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"Saved to {path}")


# ==================== RQ2 Task 1: Overall Reason Distribution ====================

# Mapping from 12 reasons to 4 categories (indices 0-2, 3-4, 5-7, 8-11)
REASON_CATEGORY_MAP = {
    0: 0, 1: 0, 2: 0,      # Category 0: Code activity related
    3: 1, 4: 1,            # Category 1: Testing related
    5: 2, 6: 2, 7: 2,      # Category 2: Developer related
    8: 3, 9: 3, 10: 3, 11: 3  # Category 3: Maintenance & security related
}

CATEGORY_NAMES = {
    0: 'Project Meta Data',
    1: 'Test Module',
    2: 'Developer',
    3: 'Issue Management'
}

# Metric indices invalid in 2025 due to lack of 2026 data (position in 12 reasons)
# 5: core_developer_turnover(year), 7: newcomer_retention(%, year)
METRICS_INVALID_2025 = {5, 7}

# Corresponding metric names
METRICS_INVALID_2025_NAMES = {
    'core_developer_turnover(year)',
    'newcomer_retention(%, year)'
}


def get_reason_index(metric_keys: List[str], reason: str) -> int:
    """Get the index of a reason in METRIC"""
    try:
        return list(metric_keys).index(reason)
    except ValueError:
        return -1


def rq2_t1_1_occurrence_and_share(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None
) -> Dict[str, Any]:
    """
    T1.1: Calculate reason occurrence rate and share

    Occurrence p(c): proportion of alarmed samples containing reason c (sum can be >1)
    Share q(c): proportion of reason c in all label occurrences (sum = 1)

    Args:
        years_alarm: rq1_dict['years_alarm'], structure: {year: {repo: {'ossf_score': float, 'our_triggered': list}}}
        metric_keys: list of METRIC.keys(), 12 reasons
        years: list of years to analyze, None means all

    Returns:
        Statistics for 12 reasons and 4 categories including occurrence and share
    """
    if years is None:
        years = list(years_alarm.keys())

    metric_keys = list(metric_keys)
    n_reasons = len(metric_keys)

    # Statistics variables
    total_alarmed = 0  # Number of alarmed samples
    total_labels = 0   # Total label count
    reason_counts = np.zeros(n_reasons)  # Count for each reason
    category_occurrence_counts = np.zeros(4)  # Sample count containing each category
    category_label_counts = np.zeros(4)  # Label count for each category

    for year in years:
        if year not in years_alarm:
            continue
        for repo, data in years_alarm[year].items():
            triggered = data.get('our_triggered', [])
            if not triggered:
                continue

            total_alarmed += 1
            total_labels += len(triggered)

            # Count each reason
            categories_hit = set()
            for reason in triggered:
                idx = get_reason_index(metric_keys, reason)
                if idx >= 0:
                    reason_counts[idx] += 1
                    cat = REASON_CATEGORY_MAP.get(idx, -1)
                    if cat >= 0:
                        categories_hit.add(cat)
                        category_label_counts[cat] += 1

            # Count category occurrence (each category counted at most once)
            for cat in categories_hit:
                category_occurrence_counts[cat] += 1

    # Calculate occurrence and share
    result = {
        'total_alarmed': int(total_alarmed),
        'total_labels': int(total_labels),
        'reasons': {},
        'categories': {}
    }

    # Statistics for 12 reasons
    for idx, reason in enumerate(metric_keys):
        count = int(reason_counts[idx])
        occurrence = count / total_alarmed if total_alarmed > 0 else 0
        share = count / total_labels if total_labels > 0 else 0
        # Wilson CI for occurrence
        ci_low, ci_high = _wilson_ci(count, total_alarmed)
        result['reasons'][reason] = {
            'count': count,
            'occurrence': occurrence,
            'share': share,
            'occurrence_ci': (ci_low, ci_high),
            'Dimension': REASON_CATEGORY_MAP.get(idx, -1)
        }

    # Statistics for 4 categories
    for cat_idx in range(4):
        occ_count = int(category_occurrence_counts[cat_idx])
        label_count = int(category_label_counts[cat_idx])
        occurrence = occ_count / total_alarmed if total_alarmed > 0 else 0
        share = label_count / total_labels if total_labels > 0 else 0
        ci_low, ci_high = _wilson_ci(occ_count, total_alarmed)
        result['categories'][CATEGORY_NAMES[cat_idx]] = {
            'occurrence_count': occ_count,
            'label_count': label_count,
            'occurrence': occurrence,
            'share': share,
            'occurrence_ci': (ci_low, ci_high)
        }

    return result


def rq2_t1_2_multi_reason_stats(
    years_alarm: Dict[int, Dict[str, Dict]],
    years: List[int] = None
) -> Dict[str, Any]:
    """
    T1.2: Multi-reason alarm statistics

    Statistics for K (number of reasons per record) distribution, including:
    - Non-alarm ratio Pr(K=0)
    - Multi-reason alarm ratio Pr(K>=2 | A=1)
    - Mean, median, P90 of K, etc.

    Args:
        years_alarm: rq1_dict['years_alarm']
        years: list of years to analyze

    Returns:
        K distribution statistics
    """
    if years is None:
        years = list(years_alarm.keys())

    k_values = []  # K values for all samples
    k_alarmed = []  # K values for alarmed samples only

    for year in years:
        if year not in years_alarm:
            continue
        for repo, data in years_alarm[year].items():
            triggered = data.get('our_triggered', [])
            k = len(triggered)
            k_values.append(k)
            if k > 0:
                k_alarmed.append(k)

    k_values = np.array(k_values)
    k_alarmed = np.array(k_alarmed) if k_alarmed else np.array([])

    total = len(k_values)
    n_no_alarm = np.sum(k_values == 0)
    n_alarmed = len(k_alarmed)
    n_multi = np.sum(k_alarmed >= 2) if len(k_alarmed) > 0 else 0

    result = {
        'total_samples': int(total),
        'n_no_alarm': int(n_no_alarm),
        'n_alarmed': int(n_alarmed),
        'n_multi_reason': int(n_multi),
        'pr_no_alarm': n_no_alarm / total if total > 0 else 0,
        'pr_multi_given_alarm': n_multi / n_alarmed if n_alarmed > 0 else 0,
        'k_distribution': {
            int(k): int(np.sum(k_values == k)) for k in range(int(k_values.max()) + 1)
        } if len(k_values) > 0 else {},
        'k_stats_all': {
            'mean': float(np.mean(k_values)) if len(k_values) > 0 else 0,
            'median': float(np.median(k_values)) if len(k_values) > 0 else 0,
            'p90': float(np.percentile(k_values, 90)) if len(k_values) > 0 else 0,
            'max': int(np.max(k_values)) if len(k_values) > 0 else 0
        },
        'k_stats_alarmed': {
            'mean': float(np.mean(k_alarmed)) if len(k_alarmed) > 0 else 0,
            'median': float(np.median(k_alarmed)) if len(k_alarmed) > 0 else 0,
            'p90': float(np.percentile(k_alarmed, 90)) if len(k_alarmed) > 0 else 0,
            'max': int(np.max(k_alarmed)) if len(k_alarmed) > 0 else 0
        }
    }

    return result


def rq2_t1_3_distribution_distance(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None
) -> Dict[str, Any]:
    """
    T1.3: Year distribution vectors and drift metrics

    Construct share distribution vector q_y for each year, calculate JSD and TV distance

    Args:
        years_alarm: rq1_dict['years_alarm']
        metric_keys: list of METRIC.keys()
        years: list of years

    Returns:
        Share vectors for each year and JSD/TV distances between year pairs
    """
    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)
    n_reasons = len(metric_keys)

    def compute_share_vector(year_data: Dict[str, Dict]) -> np.ndarray:
        """Calculate share vector for a year (excludes 2025 invalid metrics: CDT and NR)"""
        counts = np.zeros(n_reasons)
        total_labels = 0
        for repo, data in year_data.items():
            triggered = data.get('our_triggered', [])
            for reason in triggered:
                idx = get_reason_index(metric_keys, reason)
                # # Exclude 2025 invalid metrics (indices 5 and 7)
                # if idx >= 0 and idx not in METRICS_INVALID_2025:
                if idx >= 0:
                    counts[idx] += 1
                    total_labels += 1
        if total_labels > 0:
            return counts / total_labels
        return counts

    def jsd(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon Divergence"""
        # Avoid log(0)
        eps = 1e-12
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        m = 0.5 * (p + q)

        def kl(a, b):
            return np.sum(a * np.log(a / b))

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    def tv(p: np.ndarray, q: np.ndarray) -> float:
        """Total Variation Distance"""
        return 0.5 * np.sum(np.abs(p - q))

    # Calculate share vector for each year
    share_vectors = {}
    for year in years:
        if year in years_alarm:
            share_vectors[year] = compute_share_vector(years_alarm[year])

    # Calculate distance between year pairs
    year_pairs = list(combinations(sorted(share_vectors.keys()), 2))
    distances = {}
    for y1, y2 in year_pairs:
        pair_key = f"{y1}-{y2}"
        distances[pair_key] = {
            'jsd': float(jsd(share_vectors[y1], share_vectors[y2])),
            'tv': float(tv(share_vectors[y1], share_vectors[y2]))
        }

    result = {
        'share_vectors': {
            year: {metric_keys[i]: float(v) for i, v in enumerate(vec)}
            for year, vec in share_vectors.items()
        },
        'distances': distances,
        'metric_keys': metric_keys
    }

    return result


def rq2_t1_4_permutation_test(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    year_pairs: List[Tuple[int, int]] = None,
    n_permutations: int = 1000,
    distance_metric: str = 'jsd',
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    T1.4: Drift significance test (permutation test)

    Perform permutation test on year pairs to test if share distributions differ significantly

    Args:
        years_alarm: rq1_dict['years_alarm']
        metric_keys: list of METRIC.keys()
        year_pairs: list of year pairs to compare, None means all combinations
        n_permutations: number of permutations
        distance_metric: 'jsd' or 'tv'
        random_seed: random seed

    Returns:
        Permutation test results for each year pair
    """
    np.random.seed(random_seed)
    metric_keys = list(metric_keys)
    n_reasons = len(metric_keys)

    years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])
    if year_pairs is None:
        year_pairs = list(combinations(years, 2))

    def extract_records(year_data: Dict[str, Dict]) -> List[List[int]]:
        """Extract alarm records (each record is a list of reason indices, excludes 2025 invalid metrics)"""
        records = []
        for repo, data in year_data.items():
            triggered = data.get('our_triggered', [])
            if triggered:
                indices = [get_reason_index(metric_keys, r) for r in triggered]
                # # Exclude 2025 invalid metrics (indices 5 and 7)
                # indices = [i for i in indices if i >= 0 and i not in METRICS_INVALID_2025]
                indices = [i for i in indices if i >= 0]
                if indices:
                    records.append(indices)
        return records

    def records_to_share(records: List[List[int]]) -> np.ndarray:
        """Convert records to share vector"""
        counts = np.zeros(n_reasons)
        for rec in records:
            for idx in rec:
                # # Skip invalid metrics (although extract_records already filters, this is extra safety)
                # if idx not in METRICS_INVALID_2025:
                #     counts[idx] += 1
                counts[idx] += 1
        total = counts.sum()
        return counts / total if total > 0 else counts

    def compute_distance(p: np.ndarray, q: np.ndarray) -> float:
        eps = 1e-12
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        if distance_metric == 'jsd':
            m = 0.5 * (p + q)
            def kl(a, b):
                return np.sum(a * np.log(a / b))
            return 0.5 * kl(p, m) + 0.5 * kl(q, m)
        else:  # tv
            return 0.5 * np.sum(np.abs(p - q))

    results = {}
    for y1, y2 in year_pairs:
        if y1 not in years_alarm or y2 not in years_alarm:
            continue

        records_1 = extract_records(years_alarm[y1])
        records_2 = extract_records(years_alarm[y2])

        if not records_1 or not records_2:
            continue

        # Observed distance
        share_1 = records_to_share(records_1)
        share_2 = records_to_share(records_2)
        d_obs = compute_distance(share_1, share_2)

        # Permutation test
        combined = records_1 + records_2
        n1 = len(records_1)
        n_extreme = 0

        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_1 = combined[:n1]
            perm_2 = combined[n1:]
            perm_share_1 = records_to_share(perm_1)
            perm_share_2 = records_to_share(perm_2)
            d_perm = compute_distance(perm_share_1, perm_share_2)
            if d_perm >= d_obs:
                n_extreme += 1

        p_value = (1 + n_extreme) / (1 + n_permutations)

        pair_key = f"{y1}-{y2}"
        results[pair_key] = {
            'observed_distance': float(d_obs),
            'p_value': float(p_value),
            'n_permutations': n_permutations,
            'n1': len(records_1),
            'n2': len(records_2),
            'significant_0.05': p_value < 0.05,
            'significant_0.01': p_value < 0.01
        }

    return {
        'distance_metric': distance_metric,
        'results': results
    }


def rq2_t1_heatmap(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None,
    value_type: str = 'occurrence',
    fig_path: str = '',
    figsize: Tuple[float, float] = (14, 4.5),
    tick_fontsize: int = 18,
    label_fontsize: int = 18,
    annot_fontsize: int = 18,
    cmap: str = 'Blues',
    short_names: Dict[str, str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot year×reason heatmap

    Rows: each year + Total (combined)
    Columns: 12 reasons (grouped by 4 categories)
    Cells: occurrence or share

    Args:
        years_alarm: rq1_dict['years_alarm']
        metric_keys: list of METRIC.keys()
        years: list of years
        value_type: 'occurrence' or 'share'
        fig_path: save path
        figsize: figure size
        short_names: short name mapping for reasons, used for display

    Returns:
        (fig, data_matrix)
    """
    # if os.path.exists(fig_path):
    #     return 0

    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)
    n_reasons = len(metric_keys)

    # Default short names
    if short_names is None:
        short_names = {
            'code_churn_rate(%, 3week)': 'CCR',
            'burstiness(year)': 'BST',
            'commit_frequency(year)': 'CF',
            'test_proportion(%, year)': 'TP',
            'test_code_churn_ratio(3week)': 'TER',
            'core_developer_turnover(year)': 'CDT',
            'truck_factor(year)': 'TF',
            'newcomer_retention(%, year)': 'NR',
            'issue_response_efficiency(day, year)': 'IRE',
            'backlog_management_index(%, year)': 'BMI',
            'cve_exposure_increase(year)': 'CET',
            'cve_severity(year)': 'CS'
        }

    def compute_stats_for_years(year_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate occurrence and share vectors for specified years"""
        reason_counts = np.zeros(n_reasons)
        total_alarmed = 0
        total_labels = 0

        for year in year_list:
            if year not in years_alarm:
                continue
            for repo, data in years_alarm[year].items():
                triggered = data.get('our_triggered', [])
                if not triggered:
                    continue
                total_alarmed += 1
                total_labels += len(triggered)
                for reason in triggered:
                    idx = get_reason_index(metric_keys, reason)
                    if idx >= 0:
                        reason_counts[idx] += 1

        occurrence = reason_counts / total_alarmed if total_alarmed > 0 else reason_counts
        share = reason_counts / total_labels if total_labels > 0 else reason_counts
        return occurrence, share

    # Build data matrix: rows=years+Total, columns=12 reasons
    row_labels = [str(y) for y in years] + ['Total']
    n_rows = len(row_labels)
    data_matrix = np.zeros((n_rows, n_reasons))

    for i, year in enumerate(years):
        occ, shr = compute_stats_for_years([year])
        data_matrix[i, :] = occ if value_type == 'occurrence' else shr

    # Total row
    occ_total, shr_total = compute_stats_for_years(years)
    data_matrix[-1, :] = occ_total if value_type == 'occurrence' else shr_total

    # # For metrics invalid in 2025 (CDT and NR), Total row uses 2023-2024 average
    # years_without_2025 = [y for y in years if y != 2025]
    # if 2025 in years and len(years_without_2025) > 0:
    #     occ_partial, shr_partial = compute_stats_for_years(years_without_2025)
    #     for idx in METRICS_INVALID_2025:
    #         if idx < n_reasons:
    #             data_matrix[-1, idx] = occ_partial[idx] if value_type == 'occurrence' else shr_partial[idx]

    # Column labels (abbreviations)
    col_labels = [short_names.get(m, m[:10]) for m in metric_keys]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Draw heatmap using imshow
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(n_reasons))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, fontsize=tick_fontsize, rotation=45, ha='right')
    ax.set_yticklabels(row_labels, fontsize=tick_fontsize)

    # Add value annotations
    for i in range(n_rows):
        for j in range(n_reasons):
            val = data_matrix[i, j]
            text = f'{val:.2f}' if value_type == 'occurrence' else f'{val:.2f}'
            color = 'white' if val > data_matrix.max() * 0.6 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=annot_fontsize, color=color)

    # Add category separator lines
    category_boundaries = [2.5, 4.5, 7.5]  # 0-2, 3-4, 5-7, 8-11
    for boundary in category_boundaries:
        ax.axvline(x=boundary, color='white', linewidth=2)

    # Add category labels
    category_positions = [1, 3.5, 6, 9.5]
    category_labels = ['Project Meta Data', 'Test Module', 'Developer', 'Issue Management']

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar_label = 'Occurrence Rate' if value_type == 'occurrence' else 'Share'
    cbar.set_label(cbar_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    ax.set_xlabel('Metric', fontsize=label_fontsize)
    ax.set_ylabel('Year', fontsize=label_fontsize)

    title = f'Reason {"Occurrence" if value_type == "occurrence" else "Share"} by Year'
    # ax.set_title(title, fontsize=label_fontsize + 2)

    plt.tight_layout()


    fig.savefig(fig_path)
    print(f"Saved to {fig_path}")

    return fig, data_matrix


def rq2_t1_category_heatmap(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None,
    value_type: str = 'occurrence',
    fig_path: str = '',
    figsize: Tuple[float, float] = (8, 4),
    tick_fontsize: int = 18,
    label_fontsize: int = 18,
    annot_fontsize: int = 18,
    cmap: str = 'Blues'
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot year×4-category heatmap

    Rows: each year + Total
    Columns: 4 categories

    Args:
        years_alarm: rq1_dict['years_alarm']
        metric_keys: list of METRIC.keys()
        years: list of years
        value_type: 'occurrence' or 'share'

    Returns:
        (fig, data_matrix)
    """

    # if os.path.exists(fig_path):
    #     return 0
    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)

    def compute_category_stats(year_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate 4-category occurrence and share for specified years"""
        category_occurrence_counts = np.zeros(4)
        category_label_counts = np.zeros(4)
        total_alarmed = 0
        total_labels = 0

        for year in year_list:
            if year not in years_alarm:
                continue
            for repo, data in years_alarm[year].items():
                triggered = data.get('our_triggered', [])
                if not triggered:
                    continue
                total_alarmed += 1
                total_labels += len(triggered)

                categories_hit = set()
                for reason in triggered:
                    idx = get_reason_index(metric_keys, reason)
                    if idx >= 0:
                        cat = REASON_CATEGORY_MAP.get(idx, -1)
                        if cat >= 0:
                            categories_hit.add(cat)
                            category_label_counts[cat] += 1

                for cat in categories_hit:
                    category_occurrence_counts[cat] += 1

        occurrence = category_occurrence_counts / total_alarmed if total_alarmed > 0 else category_occurrence_counts
        share = category_label_counts / total_labels if total_labels > 0 else category_label_counts
        return occurrence, share

    row_labels = [str(y) for y in years] + ['Total']
    n_rows = len(row_labels)
    data_matrix = np.zeros((n_rows, 4))

    for i, year in enumerate(years):
        occ, shr = compute_category_stats([year])
        data_matrix[i, :] = occ if value_type == 'occurrence' else shr

    occ_total, shr_total = compute_category_stats(years)
    data_matrix[-1, :] = occ_total if value_type == 'occurrence' else shr_total

    # Developer category (index=2) contains CDT and NR (invalid in 2025), Total row uses 2023-2024 average
    years_without_2025 = [y for y in years if y != 2025]
    if 2025 in years and len(years_without_2025) > 0:
        occ_partial, shr_partial = compute_category_stats(years_without_2025)
        data_matrix[-1, 2] = occ_partial[2] if value_type == 'occurrence' else shr_partial[2]

    col_labels = [CATEGORY_NAMES[i] for i in range(4)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto')

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, fontsize=tick_fontsize, rotation=30, ha='right')
    ax.set_yticklabels(row_labels, fontsize=tick_fontsize)

    for i in range(n_rows):
        for j in range(4):
            val = data_matrix[i, j]
            text = f'{val:.2f}' if value_type == 'occurrence' else f'{val:.2f}'
            color = 'white' if val > data_matrix.max() * 0.6 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=annot_fontsize, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar_label = 'Occurrence Rate' if value_type == 'occurrence' else 'Share'
    cbar.set_label(cbar_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    ax.set_xlabel('Dimension', fontsize=label_fontsize)
    ax.set_ylabel('Year', fontsize=label_fontsize)

    title = f'Category {"Occurrence" if value_type == "occurrence" else "Share"} by Year'
    # ax.set_title(title, fontsize=label_fontsize + 2)

    plt.tight_layout()

    fig.savefig(fig_path)
    print(f"Saved to {fig_path}")


# ==================== RQ2 Task 2: High-score repos vs Low-score repos ====================

def rq2_t2_split_high_low(
    years_alarm: Dict[int, Dict[str, Dict]],
    threshold_high: float = 5.0,
    threshold_low: float = 5.0
) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """
    Split years_alarm into High and Low groups by OSSF score threshold

    Args:
        years_alarm: {year: {repo: {'ossf_score': float, 'our_triggered': list}}}
        threshold_high: high score threshold, >= threshold_high for High group
        threshold_low: low score threshold, < threshold_low for Low group

    Returns:
        (high_group, low_group), same structure as years_alarm
        Note: if threshold_high != threshold_low, repos in between belong to neither group
    """
    high_group = {}
    low_group = {}

    for year, repos in years_alarm.items():
        high_group[year] = {}
        low_group[year] = {}
        for repo, data in repos.items():
            score = data.get('ossf_score')
            if score is None:
                continue
            if score >= threshold_high:
                high_group[year][repo] = data
            elif score < threshold_low:
                low_group[year][repo] = data

    return high_group, low_group


def rq2_t2_group_stats(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None,
    threshold_high: float = 5.0,
    threshold_low: float = 5.0
) -> Dict[str, Any]:
    """
    T3.1: Descriptive statistics by group and year

    For each year y and group g∈{High, Low}, compute:
    - p_{y,g}(c): proportion of alarmed samples containing reason c (occurrence)
    - q_{y,g}(c): share of reason c among all label occurrences (share)
    Also compute statistics for 4 categories

    Args:
        years_alarm: rq1_dict['years_alarm']
        metric_keys: METRIC.keys()
        years: list of years
        threshold_high: high score threshold, >= threshold_high for High group
        threshold_low: low score threshold, < threshold_low for Low group

    Returns:
        dict containing statistics for High/Low groups
    """
    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)
    n_reasons = len(metric_keys)

    high_group, low_group = rq2_t2_split_high_low(years_alarm, threshold_high, threshold_low)

    def compute_stats(group_data: Dict, year_list: List[int]) -> Dict:
        """Calculate statistics for specified group"""
        reason_counts = np.zeros(n_reasons)
        category_occ_counts = np.zeros(4)
        category_label_counts = np.zeros(4)
        total_alarmed = 0
        total_labels = 0

        for year in year_list:
            if year not in group_data:
                continue
            for repo, data in group_data[year].items():
                triggered = data.get('our_triggered', [])
                if not triggered:
                    continue
                total_alarmed += 1
                total_labels += len(triggered)

                category_flags = [False] * 4
                for reason in triggered:
                    idx = get_reason_index(metric_keys, reason)
                    if idx >= 0:
                        reason_counts[idx] += 1
                        cat_idx = idx // 3 if idx < 9 else 3
                        if idx < 3:
                            cat_idx = 0
                        elif idx < 5:
                            cat_idx = 1
                        elif idx < 8:
                            cat_idx = 2
                        else:
                            cat_idx = 3
                        category_flags[cat_idx] = True
                        category_label_counts[cat_idx] += 1

                for i, flag in enumerate(category_flags):
                    if flag:
                        category_occ_counts[i] += 1

        occurrence = reason_counts / total_alarmed if total_alarmed > 0 else reason_counts
        share = reason_counts / total_labels if total_labels > 0 else reason_counts
        cat_occ = category_occ_counts / total_alarmed if total_alarmed > 0 else category_occ_counts
        cat_share = category_label_counts / total_labels if total_labels > 0 else category_label_counts

        return {
            'total_alarmed': int(total_alarmed),
            'total_labels': int(total_labels),
            'reason_counts': reason_counts.tolist(),
            'occurrence': occurrence.tolist(),
            'share': share.tolist(),
            'category_occurrence': cat_occ.tolist(),
            'category_share': cat_share.tolist()
        }

    result = {
        'threshold_high': threshold_high,
        'threshold_low': threshold_low,
        'years': years,
        'by_year': {},
        'total': {}
    }

    # Per-year statistics
    for year in years:
        result['by_year'][year] = {
            'high': compute_stats(high_group, [year]),
            'low': compute_stats(low_group, [year])
        }

    # Aggregated statistics
    result['total'] = {
        'high': compute_stats(high_group, years),
        'low': compute_stats(low_group, years)
    }

    # # For metrics invalid in 2025, Total uses 2023-2024 average
    # years_without_2025 = [y for y in years if y != 2025]
    # if 2025 in years and len(years_without_2025) > 0:
    #     for group_name in ['high', 'low']:
    #         group_data = high_group if group_name == 'high' else low_group
    #         partial_stats = compute_stats(group_data, years_without_2025)
    #         # Replace invalid metrics' occurrence and share
    #         for idx in METRICS_INVALID_2025:
    #             if idx < n_reasons:
    #                 result['total'][group_name]['occurrence'][idx] = partial_stats['occurrence'][idx]
    #                 result['total'][group_name]['share'][idx] = partial_stats['share'][idx]
    #         # Replace Developer category (index=2) occurrence and share
    #         result['total'][group_name]['category_occurrence'][2] = partial_stats['category_occurrence'][2]
    #         result['total'][group_name]['category_share'][2] = partial_stats['category_share'][2]

    return result


def rq2_t2_group_difference_test(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None,
    threshold_high: float = 5.0,
    threshold_low: float = 5.0,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    T3.2: Group difference test (per-reason + effect size + FDR)

    For fixed year y and reason c, construct 2×2 contingency table (within alarmed samples):
                 X^(c)=1  X^(c)=0
        High       a        b
        Low        c        d

    Perform chi-square or Fisher's exact test, report:
    - Risk difference RD = p_High - p_Low
    - Odds ratio OR = (a/b) / (c/d)
    Apply FDR correction for 12 reasons and 4 categories

    Args:
        years_alarm: rq1_dict['years_alarm']
        metric_keys: METRIC.keys()
        years: list of years
        threshold_high: high score threshold, >= threshold_high for High group
        threshold_low: low score threshold, < threshold_low for Low group
        alpha: significance level

    Returns:
        dict containing test results
    """
    from scipy.stats import chi2_contingency, fisher_exact
    from statsmodels.stats.multitest import multipletests

    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)

    high_group, low_group = rq2_t2_split_high_low(years_alarm, threshold_high, threshold_low)

    def count_reason_in_group(group_data: Dict, year_list: List[int], reason: str) -> Tuple[int, int]:
        """Count presence/absence of reason in alarmed samples within group"""
        has_reason = 0
        no_reason = 0
        for year in year_list:
            if year not in group_data:
                continue
            for repo, data in group_data[year].items():
                triggered = data.get('our_triggered', [])
                if not triggered:
                    continue
                if reason in triggered:
                    has_reason += 1
                else:
                    no_reason += 1
        return has_reason, no_reason

    def count_category_in_group(group_data: Dict, year_list: List[int], cat_idx: int) -> Tuple[int, int]:
        """Count presence/absence of category in alarmed samples within group"""
        # Category index: 0=Code Activity(0-2), 1=Testing(3-4), 2=Developer(5-7), 3=Maint.&Sec.(8-11)
        cat_ranges = [(0, 3), (3, 5), (5, 8), (8, 12)]
        cat_reasons = [list(metric_keys)[i] for i in range(cat_ranges[cat_idx][0], cat_ranges[cat_idx][1])]

        has_cat = 0
        no_cat = 0
        for year in year_list:
            if year not in group_data:
                continue
            for repo, data in group_data[year].items():
                triggered = data.get('our_triggered', [])
                if not triggered:
                    continue
                if any(r in triggered for r in cat_reasons):
                    has_cat += 1
                else:
                    no_cat += 1
        return has_cat, no_cat

    def run_test(a, b, c, d) -> Dict:
        """Run chi-square or Fisher's exact test"""
        table = np.array([[a, b], [c, d]])
        n_total = a + b + c + d

        # Calculate effect size
        p_high = a / (a + b) if (a + b) > 0 else 0
        p_low = c / (c + d) if (c + d) > 0 else 0
        rd = p_high - p_low

        # OR
        if b > 0 and c > 0 and d > 0:
            or_val = (a * d) / (b * c) if (b * c) > 0 else float('inf')
        else:
            or_val = float('nan')

        # Select test method
        min_expected = min(
            (a + b) * (a + c) / n_total if n_total > 0 else 0,
            (a + b) * (b + d) / n_total if n_total > 0 else 0,
            (c + d) * (a + c) / n_total if n_total > 0 else 0,
            (c + d) * (b + d) / n_total if n_total > 0 else 0
        ) if n_total > 0 else 0

        if min_expected < 5 or n_total < 20:
            # Fisher's exact test
            try:
                _, p_val = fisher_exact(table)
                test_method = 'fisher'
            except:
                p_val = 1.0
                test_method = 'error'
        else:
            # Chi-square test
            try:
                chi2, p_val, dof, expected = chi2_contingency(table)
                test_method = 'chi2'
            except:
                p_val = 1.0
                test_method = 'error'

        return {
            'contingency_table': [[a, b], [c, d]],
            'p_high': p_high,
            'p_low': p_low,
            'risk_difference': rd,
            'odds_ratio': or_val,
            'p_value': p_val,
            'test_method': test_method
        }

    result = {
        'threshold_high': threshold_high,
        'threshold_low': threshold_low,
        'alpha': alpha,
        'by_year': {},
        'total': {}
    }

    # Per-year tests
    for year in years:
        year_result = {'reasons': {}, 'categories': {}}

        # 12 reasons
        for reason in metric_keys:
            a, b = count_reason_in_group(high_group, [year], reason)
            c, d = count_reason_in_group(low_group, [year], reason)
            year_result['reasons'][reason] = run_test(a, b, c, d)

        # 4 categories
        cat_names = ['Project Meta Data', 'Test Module', 'Developer', 'Issue Management']
        for cat_idx, cat_name in enumerate(cat_names):
            a, b = count_category_in_group(high_group, [year], cat_idx)
            c, d = count_category_in_group(low_group, [year], cat_idx)
            year_result['categories'][cat_name] = run_test(a, b, c, d)

        # FDR correction
        reason_pvals = [year_result['reasons'][r]['p_value'] for r in metric_keys]
        cat_pvals = [year_result['categories'][c]['p_value'] for c in cat_names]
        all_pvals = reason_pvals + cat_pvals

        if len(all_pvals) > 0 and not all(np.isnan(all_pvals)):
            _, fdr_corrected, _, _ = multipletests(all_pvals, alpha=alpha, method='fdr_bh')
            for i, reason in enumerate(metric_keys):
                year_result['reasons'][reason]['p_fdr'] = fdr_corrected[i]
                year_result['reasons'][reason]['significant_fdr'] = fdr_corrected[i] < alpha
            for i, cat_name in enumerate(cat_names):
                year_result['categories'][cat_name]['p_fdr'] = fdr_corrected[len(metric_keys) + i]
                year_result['categories'][cat_name]['significant_fdr'] = fdr_corrected[len(metric_keys) + i] < alpha

        result['by_year'][year] = year_result

    # Aggregated tests
    total_result = {'reasons': {}, 'categories': {}}

    for reason in metric_keys:
        a, b = count_reason_in_group(high_group, years, reason)
        c, d = count_reason_in_group(low_group, years, reason)
        total_result['reasons'][reason] = run_test(a, b, c, d)

    cat_names = ['Project Meta Data', 'Test Module', 'Developer', 'Issue Management']
    for cat_idx, cat_name in enumerate(cat_names):
        a, b = count_category_in_group(high_group, years, cat_idx)
        c, d = count_category_in_group(low_group, years, cat_idx)
        total_result['categories'][cat_name] = run_test(a, b, c, d)

    # FDR correction
    reason_pvals = [total_result['reasons'][r]['p_value'] for r in metric_keys]
    cat_pvals = [total_result['categories'][c]['p_value'] for c in cat_names]
    all_pvals = reason_pvals + cat_pvals

    if len(all_pvals) > 0 and not all(np.isnan(all_pvals)):
        _, fdr_corrected, _, _ = multipletests(all_pvals, alpha=alpha, method='fdr_bh')
        for i, reason in enumerate(metric_keys):
            total_result['reasons'][reason]['p_fdr'] = fdr_corrected[i]
            total_result['reasons'][reason]['significant_fdr'] = fdr_corrected[i] < alpha
        for i, cat_name in enumerate(cat_names):
            total_result['categories'][cat_name]['p_fdr'] = fdr_corrected[len(metric_keys) + i]
            total_result['categories'][cat_name]['significant_fdr'] = fdr_corrected[len(metric_keys) + i] < alpha

    result['total'] = total_result

    return result


def rq2_t2_group_heatmap(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None,
    threshold_high: float = 5.0,
    threshold_low: float = 5.0,
    value_type: str = 'occurrence',
    fig_path: str = '',
    figsize: Tuple[float, float] = (16, 4.5),
    tick_fontsize: int = 16,
    label_fontsize: int = 16,
    annot_fontsize: int = 16,
    cmap: str = 'viridis',
    short_names: Dict[str, str] = None
) -> Tuple[plt.Figure, Dict]:
    """
    Plot heatmap for High/Low groups (side by side)

    Args:
        years_alarm: rq1_dict['years_alarm']
        metric_keys: METRIC.keys()
        years: list of years
        threshold_high: high score threshold, >= threshold_high for High group
        threshold_low: low score threshold, < threshold_low for Low group
        value_type: 'occurrence' or 'share'
        fig_path: save path

    Returns:
        (fig, data_dict)
    """
    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)
    n_reasons = len(metric_keys)

    if short_names is None:
        short_names = {
            'code_churn_rate(%, 3week)': 'CCR',
            'burstiness(year)': 'BST',
            'commit_frequency(year)': 'CF',
            'test_proportion(%, year)': 'TP',
            'test_code_churn_ratio(3week)': 'TER',
            'core_developer_turnover(year)': 'CDT',
            'truck_factor(year)': 'TF',
            'newcomer_retention(%, year)': 'NR',
            'issue_response_efficiency(day, year)': 'IRE',
            'backlog_management_index(%, year)': 'BMI',
            'cve_exposure_increase(year)': 'CET',
            'cve_severity(year)': 'CS'
        }

    # Get group statistics
    stats = rq2_t2_group_stats(years_alarm, metric_keys, years, threshold_high, threshold_low)

    # Build data matrix
    row_labels = [str(y) for y in years] + ['Total']
    n_rows = len(row_labels)
    col_labels = [short_names.get(m, m[:10]) for m in metric_keys]

    high_matrix = np.zeros((n_rows, n_reasons))
    low_matrix = np.zeros((n_rows, n_reasons))

    for i, year in enumerate(years):
        high_data = stats['by_year'][year]['high']
        low_data = stats['by_year'][year]['low']
        high_matrix[i, :] = high_data['occurrence'] if value_type == 'occurrence' else high_data['share']
        low_matrix[i, :] = low_data['occurrence'] if value_type == 'occurrence' else low_data['share']

    high_total = stats['total']['high']
    low_total = stats['total']['low']
    high_matrix[-1, :] = high_total['occurrence'] if value_type == 'occurrence' else high_total['share']
    low_matrix[-1, :] = low_total['occurrence'] if value_type == 'occurrence' else low_total['share']

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    vmin = min(high_matrix.min(), low_matrix.min())
    vmax = max(high_matrix.max(), low_matrix.max())

    # High group heatmap
    im1 = axes[0].imshow(high_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_xticks(np.arange(n_reasons))
    axes[0].set_yticks(np.arange(n_rows))
    axes[0].set_xticklabels(col_labels, fontsize=tick_fontsize, rotation=45, ha='right')
    axes[0].set_yticklabels(row_labels, fontsize=tick_fontsize)
    axes[0].set_title(f'High Score (≥{threshold_high})', fontsize=label_fontsize)
    axes[0].set_ylabel('Year', fontsize=label_fontsize)
    axes[0].set_xlabel('Metric', fontsize=label_fontsize)

    for i in range(n_rows):
        for j in range(n_reasons):
            val = high_matrix[i, j]
            text = f'{val:.2f}' if value_type == 'occurrence' else f'{val:.2f}'
            color = 'white' if val > vmax * 0.6 else 'black'
            axes[0].text(j, i, text, ha='center', va='center', fontsize=annot_fontsize, color=color)

    # Category separator lines
    for boundary in [2.5, 4.5, 7.5]:
        axes[0].axvline(x=boundary, color='white', linewidth=2)

    # Low group heatmap
    im2 = axes[1].imshow(low_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_xticks(np.arange(n_reasons))
    axes[1].set_xticklabels(col_labels, fontsize=tick_fontsize, rotation=45, ha='right')
    axes[1].set_title(f'Low Score (<{threshold_low})', fontsize=label_fontsize)

    for i in range(n_rows):
        for j in range(n_reasons):
            val = low_matrix[i, j]
            text = f'{val:.2f}' if value_type == 'occurrence' else f'{val:.2f}'
            color = 'white' if val > vmax * 0.6 else 'black'
            axes[1].text(j, i, text, ha='center', va='center', fontsize=annot_fontsize, color=color)

    for boundary in [2.5, 4.5, 7.5]:
        axes[1].axvline(x=boundary, color='white', linewidth=2)

    # Shared colorbar - adjust position to avoid overlap
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar_label = 'Occurrence Rate' if value_type == 'occurrence' else 'Share'
    cbar.set_label(cbar_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    plt.subplots_adjust(bottom=0.2, wspace=0.1)

    if fig_path:
        fig.savefig(fig_path)
        print(f"Saved to {fig_path}")

    return fig, {'high': high_matrix, 'low': low_matrix, 'stats': stats}


def rq2_t2_group_category_heatmap(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None,
    threshold_high: float = 5.0,
    threshold_low: float = 5.0,
    value_type: str = 'occurrence',
    fig_path: str = '',
    figsize: Tuple[float, float] = (10, 4),
    tick_fontsize: int = 16,
    label_fontsize: int = 16,
    annot_fontsize: int = 16,
    cmap: str = 'viridis'
) -> Tuple[plt.Figure, Dict]:
    """
    Plot 4-category heatmap for High/Low groups (side by side)

    Args:
        years_alarm: rq1_dict['years_alarm']
        metric_keys: METRIC.keys()
        years: list of years
        threshold_high: high score threshold
        threshold_low: low score threshold
        value_type: 'occurrence' or 'share'
        fig_path: save path

    Returns:
        (fig, data_dict)
    """
    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)
    cat_names = ['Project Meta Data', 'Test Module', 'Developer', 'Issue Management']

    # Get group statistics
    stats = rq2_t2_group_stats(years_alarm, metric_keys, years, threshold_high, threshold_low)

    # Build data matrix
    row_labels = [str(y) for y in years] + ['Total']
    n_rows = len(row_labels)

    high_matrix = np.zeros((n_rows, 4))
    low_matrix = np.zeros((n_rows, 4))

    for i, year in enumerate(years):
        high_data = stats['by_year'][year]['high']
        low_data = stats['by_year'][year]['low']
        high_matrix[i, :] = high_data['category_occurrence'] if value_type == 'occurrence' else high_data['category_share']
        low_matrix[i, :] = low_data['category_occurrence'] if value_type == 'occurrence' else low_data['category_share']

    high_total = stats['total']['high']
    low_total = stats['total']['low']
    high_matrix[-1, :] = high_total['category_occurrence'] if value_type == 'occurrence' else high_total['category_share']
    low_matrix[-1, :] = low_total['category_occurrence'] if value_type == 'occurrence' else low_total['category_share']

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    vmin = min(high_matrix.min(), low_matrix.min())
    vmax = max(high_matrix.max(), low_matrix.max())

    # High group heatmap
    im1 = axes[0].imshow(high_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_xticks(np.arange(4))
    axes[0].set_yticks(np.arange(n_rows))
    axes[0].set_xticklabels(cat_names, fontsize=tick_fontsize, rotation=30, ha='right')
    axes[0].set_yticklabels(row_labels, fontsize=tick_fontsize)
    axes[0].set_title(f'High Score (≥{threshold_high})', fontsize=label_fontsize)
    axes[0].set_ylabel('Year', fontsize=label_fontsize)
    axes[0].set_xlabel('Dimension', fontsize=label_fontsize)

    for i in range(n_rows):
        for j in range(4):
            val = high_matrix[i, j]
            text = f'{val:.2f}' if value_type == 'occurrence' else f'{val:.2f}'
            color = 'white' if val > vmax * 0.6 else 'black'
            axes[0].text(j, i, text, ha='center', va='center', fontsize=annot_fontsize, color=color)

    # Low group heatmap
    im2 = axes[1].imshow(low_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels(cat_names, fontsize=tick_fontsize, rotation=30, ha='right')
    axes[1].set_title(f'Low Score (<{threshold_low})', fontsize=label_fontsize)

    for i in range(n_rows):
        for j in range(4):
            val = low_matrix[i, j]
            text = f'{val:.2f}' if value_type == 'occurrence' else f'{val:.2f}'
            color = 'white' if val > vmax * 0.6 else 'black'
            axes[1].text(j, i, text, ha='center', va='center', fontsize=annot_fontsize, color=color)

    # Shared colorbar - adjust position to avoid overlap
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar_label = 'Occurrence Rate' if value_type == 'occurrence' else 'Share'
    cbar.set_label(cbar_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    plt.subplots_adjust(bottom=0.2, wspace=0.15)

    if fig_path:
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Saved to {fig_path}")

    return fig, {'high': high_matrix, 'low': low_matrix, 'stats': stats}


# ==================== RQ2 Task 3: Cross-year consistent high-score repos vs low-score repos ====================

def rq2_t3_get_consistent_repos(
    years_alarm: Dict[int, Dict[str, Dict]],
    years: List[int],
    threshold_high: float = 5.0,
    threshold_low: float = 5.0
) -> Tuple[set, set]:
    """
    Get cross-year consistent high-score and low-score repos

    High-score repos: score >= threshold_high in all years
    Low-score repos: score < threshold_low in all years
    """
    repo_scores = {}
    for year in years:
        if year not in years_alarm:
            continue
        for repo, data in years_alarm[year].items():
            if repo not in repo_scores:
                repo_scores[repo] = {}
            repo_scores[repo][year] = data.get('ossf_score')

    high_repos = set()
    low_repos = set()

    for repo, scores in repo_scores.items():
        if len(scores) != len(years):
            continue
        if any(s is None for s in scores.values()):
            continue

        score_values = list(scores.values())
        if all(s >= threshold_high for s in score_values):
            high_repos.add(repo)
        elif all(s < threshold_low for s in score_values):
            low_repos.add(repo)

    return high_repos, low_repos


def rq2_t3_consistent_group_stats(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None,
    threshold_high: float = 5.0,
    threshold_low: float = 5.0
) -> Dict[str, Any]:
    """T3: Descriptive statistics for cross-year consistent groups"""
    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)
    n_reasons = len(metric_keys)

    high_repos, low_repos = rq2_t3_get_consistent_repos(years_alarm, years, threshold_high, threshold_low)

    def compute_stats(repo_set: set, year_list: List[int]) -> Dict:
        reason_counts = np.zeros(n_reasons)
        category_occ_counts = np.zeros(4)
        category_label_counts = np.zeros(4)
        total_alarmed = 0
        total_labels = 0

        for year in year_list:
            if year not in years_alarm:
                continue
            for repo, data in years_alarm[year].items():
                if repo not in repo_set:
                    continue
                triggered = data.get('our_triggered', [])
                if not triggered:
                    continue
                total_alarmed += 1
                total_labels += len(triggered)

                category_flags = [False] * 4
                for reason in triggered:
                    idx = get_reason_index(metric_keys, reason)
                    if idx >= 0:
                        reason_counts[idx] += 1
                        if idx < 3:
                            cat_idx = 0
                        elif idx < 5:
                            cat_idx = 1
                        elif idx < 8:
                            cat_idx = 2
                        else:
                            cat_idx = 3
                        category_flags[cat_idx] = True
                        category_label_counts[cat_idx] += 1

                for i, flag in enumerate(category_flags):
                    if flag:
                        category_occ_counts[i] += 1

        occurrence = reason_counts / total_alarmed if total_alarmed > 0 else reason_counts
        share = reason_counts / total_labels if total_labels > 0 else reason_counts
        cat_occ = category_occ_counts / total_alarmed if total_alarmed > 0 else category_occ_counts
        cat_share = category_label_counts / total_labels if total_labels > 0 else category_label_counts

        return {
            'n_repos': len(repo_set),
            'total_alarmed': int(total_alarmed),
            'total_labels': int(total_labels),
            'reason_counts': reason_counts.tolist(),
            'occurrence': occurrence.tolist(),
            'share': share.tolist(),
            'category_occurrence': cat_occ.tolist(),
            'category_share': cat_share.tolist()
        }

    result = {
        'threshold_high': threshold_high,
        'threshold_low': threshold_low,
        'years': years,
        'high_repos': list(high_repos),
        'low_repos': list(low_repos),
        'by_year': {},
        'total': {}
    }

    for year in years:
        result['by_year'][year] = {
            'high': compute_stats(high_repos, [year]),
            'low': compute_stats(low_repos, [year])
        }

    result['total'] = {
        'high': compute_stats(high_repos, years),
        'low': compute_stats(low_repos, years)
    }

    # # For metrics invalid in 2025, Total uses 2023-2024 average
    # years_without_2025 = [y for y in years if y != 2025]
    # if 2025 in years and len(years_without_2025) > 0:
    #     for group_name, repo_set in [('high', high_repos), ('low', low_repos)]:
    #         partial_stats = compute_stats(repo_set, years_without_2025)
    #         # Replace invalid metrics' occurrence and share
    #         for idx in METRICS_INVALID_2025:
    #             if idx < n_reasons:
    #                 result['total'][group_name]['occurrence'][idx] = partial_stats['occurrence'][idx]
    #                 result['total'][group_name]['share'][idx] = partial_stats['share'][idx]
    #         # Replace Developer category (index=2) occurrence and share
    #         result['total'][group_name]['category_occurrence'][2] = partial_stats['category_occurrence'][2]
    #         result['total'][group_name]['category_share'][2] = partial_stats['category_share'][2]

    return result


def rq2_t3_consistent_group_difference_test(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None,
    threshold_high: float = 5.0,
    threshold_low: float = 5.0,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """T3: Group difference test for cross-year consistent groups"""
    from scipy.stats import chi2_contingency, fisher_exact
    from statsmodels.stats.multitest import multipletests

    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)
    high_repos, low_repos = rq2_t3_get_consistent_repos(years_alarm, years, threshold_high, threshold_low)

    def count_reason_in_group(repo_set: set, year_list: List[int], reason: str) -> Tuple[int, int]:
        has_reason = 0
        no_reason = 0
        for year in year_list:
            if year not in years_alarm:
                continue
            for repo, data in years_alarm[year].items():
                if repo not in repo_set:
                    continue
                triggered = data.get('our_triggered', [])
                if not triggered:
                    continue
                if reason in triggered:
                    has_reason += 1
                else:
                    no_reason += 1
        return has_reason, no_reason

    def count_category_in_group(repo_set: set, year_list: List[int], cat_idx: int) -> Tuple[int, int]:
        cat_ranges = [(0, 3), (3, 5), (5, 8), (8, 12)]
        cat_reasons = [list(metric_keys)[i] for i in range(cat_ranges[cat_idx][0], cat_ranges[cat_idx][1])]
        has_cat = 0
        no_cat = 0
        for year in year_list:
            if year not in years_alarm:
                continue
            for repo, data in years_alarm[year].items():
                if repo not in repo_set:
                    continue
                triggered = data.get('our_triggered', [])
                if not triggered:
                    continue
                if any(r in triggered for r in cat_reasons):
                    has_cat += 1
                else:
                    no_cat += 1
        return has_cat, no_cat

    def run_test(a, b, c, d) -> Dict:
        n_total = a + b + c + d
        p_high = a / (a + b) if (a + b) > 0 else 0
        p_low = c / (c + d) if (c + d) > 0 else 0
        rd = p_high - p_low
        or_val = (a * d) / (b * c) if (b > 0 and c > 0 and (b * c) > 0) else float('nan')

        min_expected = min(
            (a + b) * (a + c) / n_total if n_total > 0 else 0,
            (a + b) * (b + d) / n_total if n_total > 0 else 0,
            (c + d) * (a + c) / n_total if n_total > 0 else 0,
            (c + d) * (b + d) / n_total if n_total > 0 else 0
        ) if n_total > 0 else 0

        table = np.array([[a, b], [c, d]])
        if min_expected < 5 or n_total < 20:
            try:
                _, p_val = fisher_exact(table)
                test_method = 'fisher'
            except:
                p_val = 1.0
                test_method = 'error'
        else:
            try:
                chi2, p_val, dof, expected = chi2_contingency(table)
                test_method = 'chi2'
            except:
                p_val = 1.0
                test_method = 'error'

        return {
            'contingency_table': [[a, b], [c, d]],
            'p_high': p_high, 'p_low': p_low,
            'risk_difference': rd, 'odds_ratio': or_val,
            'p_value': p_val, 'test_method': test_method
        }

    result = {
        'threshold_high': threshold_high,
        'threshold_low': threshold_low,
        'alpha': alpha,
        'n_high_repos': len(high_repos),
        'n_low_repos': len(low_repos),
        'by_year': {},
        'total': {}
    }

    cat_names = ['Project Meta Data', 'Test Module', 'Developer', 'Issue Management']

    for year in years:
        year_result = {'reasons': {}, 'categories': {}}
        for reason in metric_keys:
            a, b = count_reason_in_group(high_repos, [year], reason)
            c, d = count_reason_in_group(low_repos, [year], reason)
            year_result['reasons'][reason] = run_test(a, b, c, d)

        for cat_idx, cat_name in enumerate(cat_names):
            a, b = count_category_in_group(high_repos, [year], cat_idx)
            c, d = count_category_in_group(low_repos, [year], cat_idx)
            year_result['categories'][cat_name] = run_test(a, b, c, d)

        reason_pvals = [year_result['reasons'][r]['p_value'] for r in metric_keys]
        cat_pvals = [year_result['categories'][c]['p_value'] for c in cat_names]
        all_pvals = reason_pvals + cat_pvals
        if len(all_pvals) > 0 and not all(np.isnan(all_pvals)):
            _, fdr_corrected, _, _ = multipletests(all_pvals, alpha=alpha, method='fdr_bh')
            for i, reason in enumerate(metric_keys):
                year_result['reasons'][reason]['p_fdr'] = fdr_corrected[i]
                year_result['reasons'][reason]['significant_fdr'] = fdr_corrected[i] < alpha
            for i, cat_name in enumerate(cat_names):
                year_result['categories'][cat_name]['p_fdr'] = fdr_corrected[len(metric_keys) + i]
                year_result['categories'][cat_name]['significant_fdr'] = fdr_corrected[len(metric_keys) + i] < alpha

        result['by_year'][year] = year_result

    total_result = {'reasons': {}, 'categories': {}}
    for reason in metric_keys:
        a, b = count_reason_in_group(high_repos, years, reason)
        c, d = count_reason_in_group(low_repos, years, reason)
        total_result['reasons'][reason] = run_test(a, b, c, d)

    for cat_idx, cat_name in enumerate(cat_names):
        a, b = count_category_in_group(high_repos, years, cat_idx)
        c, d = count_category_in_group(low_repos, years, cat_idx)
        total_result['categories'][cat_name] = run_test(a, b, c, d)

    reason_pvals = [total_result['reasons'][r]['p_value'] for r in metric_keys]
    cat_pvals = [total_result['categories'][c]['p_value'] for c in cat_names]
    all_pvals = reason_pvals + cat_pvals
    if len(all_pvals) > 0 and not all(np.isnan(all_pvals)):
        _, fdr_corrected, _, _ = multipletests(all_pvals, alpha=alpha, method='fdr_bh')
        for i, reason in enumerate(metric_keys):
            total_result['reasons'][reason]['p_fdr'] = fdr_corrected[i]
            total_result['reasons'][reason]['significant_fdr'] = fdr_corrected[i] < alpha
        for i, cat_name in enumerate(cat_names):
            total_result['categories'][cat_name]['p_fdr'] = fdr_corrected[len(metric_keys) + i]
            total_result['categories'][cat_name]['significant_fdr'] = fdr_corrected[len(metric_keys) + i] < alpha

    result['total'] = total_result
    return result


def rq2_t3_consistent_group_heatmap(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None,
    threshold_high: float = 5.0,
    threshold_low: float = 5.0,
    value_type: str = 'occurrence',
    fig_path: str = '',
    figsize: Tuple[float, float] = (16, 4.5),
    tick_fontsize: int = 16,
    label_fontsize: int = 16,
    annot_fontsize: int = 16,
    cmap: str = 'viridis',
    short_names: Dict[str, str] = None
) -> Tuple[plt.Figure, Dict]:
    """Plot High/Low heatmap for cross-year consistent groups (12 reasons)"""
    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)
    n_reasons = len(metric_keys)

    if short_names is None:
        short_names = {
            'code_churn_rate(%, 3week)': 'CCR', 'burstiness(year)': 'BST',
            'commit_frequency(year)': 'CF', 'test_proportion(%, year)': 'TP',
            'test_code_churn_ratio(3week)': 'TER', 'core_developer_turnover(year)': 'CDT',
            'truck_factor(year)': 'TF', 'newcomer_retention(%, year)': 'NR',
            'issue_response_efficiency(day, year)': 'IRE', 'backlog_management_index(%, year)': 'BMI',
            'cve_exposure_increase(year)': 'CET', 'cve_severity(year)': 'CS'
        }

    stats = rq2_t3_consistent_group_stats(years_alarm, metric_keys, years, threshold_high, threshold_low)

    row_labels = [str(y) for y in years] + ['Total']
    n_rows = len(row_labels)
    col_labels = [short_names.get(m, m[:10]) for m in metric_keys]

    high_matrix = np.zeros((n_rows, n_reasons))
    low_matrix = np.zeros((n_rows, n_reasons))

    for i, year in enumerate(years):
        high_matrix[i, :] = stats['by_year'][year]['high']['occurrence'] if value_type == 'occurrence' else stats['by_year'][year]['high']['share']
        low_matrix[i, :] = stats['by_year'][year]['low']['occurrence'] if value_type == 'occurrence' else stats['by_year'][year]['low']['share']

    high_matrix[-1, :] = stats['total']['high']['occurrence'] if value_type == 'occurrence' else stats['total']['high']['share']
    low_matrix[-1, :] = stats['total']['low']['occurrence'] if value_type == 'occurrence' else stats['total']['low']['share']

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    vmin = min(high_matrix.min(), low_matrix.min())
    vmax = max(high_matrix.max(), low_matrix.max())
    n_high = stats['total']['high']['n_repos']
    n_low = stats['total']['low']['n_repos']

    im1 = axes[0].imshow(high_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_xticks(np.arange(n_reasons))
    axes[0].set_yticks(np.arange(n_rows))
    axes[0].set_xticklabels(col_labels, fontsize=tick_fontsize, rotation=45, ha='right')
    axes[0].set_yticklabels(row_labels, fontsize=tick_fontsize)
    axes[0].set_title(f'High Score (≥{threshold_high})', fontsize=label_fontsize)
    axes[0].set_ylabel('Year', fontsize=label_fontsize)
    axes[0].set_xlabel('Metric', fontsize=label_fontsize)

    for i in range(n_rows):
        for j in range(n_reasons):
            val = high_matrix[i, j]
            text = f'{val:.2f}' if value_type == 'occurrence' else f'{val:.2f}'
            color = 'white' if val > vmax * 0.6 else 'black'
            axes[0].text(j, i, text, ha='center', va='center', fontsize=annot_fontsize, color=color)

    for boundary in [2.5, 4.5, 7.5]:
        axes[0].axvline(x=boundary, color='white', linewidth=2)

    im2 = axes[1].imshow(low_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_xticks(np.arange(n_reasons))
    axes[1].set_xticklabels(col_labels, fontsize=tick_fontsize, rotation=45, ha='right')
    axes[1].set_title(f'Low Score (<{threshold_low})', fontsize=label_fontsize)

    for i in range(n_rows):
        for j in range(n_reasons):
            val = low_matrix[i, j]
            text = f'{val:.2f}' if value_type == 'occurrence' else f'{val:.2f}'
            color = 'white' if val > vmax * 0.6 else 'black'
            axes[1].text(j, i, text, ha='center', va='center', fontsize=annot_fontsize, color=color)

    for boundary in [2.5, 4.5, 7.5]:
        axes[1].axvline(x=boundary, color='white', linewidth=2)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Occurrence Rate' if value_type == 'occurrence' else 'Share', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    plt.subplots_adjust(bottom=0.2, wspace=0.1)

    if fig_path:
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Saved to {fig_path}")

    return fig, {'high': high_matrix, 'low': low_matrix, 'stats': stats}


def rq2_t3_consistent_group_category_heatmap(
    years_alarm: Dict[int, Dict[str, Dict]],
    metric_keys: List[str],
    years: List[int] = None,
    threshold_high: float = 5.0,
    threshold_low: float = 5.0,
    value_type: str = 'occurrence',
    fig_path: str = '',
    figsize: Tuple[float, float] = (7, 4),
    tick_fontsize: int = 16,
    label_fontsize: int = 16,
    annot_fontsize: int = 16,
    cmap: str = 'viridis'
) -> Tuple[plt.Figure, Dict]:
    """Plot High/Low heatmap for cross-year consistent groups (4 categories)"""
    if years is None:
        years = sorted([y for y in years_alarm.keys() if isinstance(y, int)])

    metric_keys = list(metric_keys)
    cat_names = ['Project Meta Data', 'Test Module', 'Developer', 'Issue Management']

    stats = rq2_t3_consistent_group_stats(years_alarm, metric_keys, years, threshold_high, threshold_low)

    row_labels = [str(y) for y in years] + ['Total']
    n_rows = len(row_labels)

    high_matrix = np.zeros((n_rows, 4))
    low_matrix = np.zeros((n_rows, 4))

    for i, year in enumerate(years):
        high_matrix[i, :] = stats['by_year'][year]['high']['category_occurrence'] if value_type == 'occurrence' else stats['by_year'][year]['high']['category_share']
        low_matrix[i, :] = stats['by_year'][year]['low']['category_occurrence'] if value_type == 'occurrence' else stats['by_year'][year]['low']['category_share']

    high_matrix[-1, :] = stats['total']['high']['category_occurrence'] if value_type == 'occurrence' else stats['total']['high']['category_share']
    low_matrix[-1, :] = stats['total']['low']['category_occurrence'] if value_type == 'occurrence' else stats['total']['low']['category_share']

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    vmin = min(high_matrix.min(), low_matrix.min())
    vmax = max(high_matrix.max(), low_matrix.max())
    n_high = stats['total']['high']['n_repos']
    n_low = stats['total']['low']['n_repos']

    im1 = axes[0].imshow(high_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_xticks(np.arange(4))
    axes[0].set_yticks(np.arange(n_rows))
    axes[0].set_xticklabels(cat_names, fontsize=tick_fontsize, rotation=30, ha='right')
    axes[0].set_yticklabels(row_labels, fontsize=tick_fontsize)
    axes[0].set_title(f'High Score (≥{threshold_high})', fontsize=label_fontsize)
    axes[0].set_ylabel('Year', fontsize=label_fontsize)
    axes[0].set_xlabel('Dimension', fontsize=label_fontsize)

    for i in range(n_rows):
        for j in range(4):
            val = high_matrix[i, j]
            text = f'{val:.2f}' if value_type == 'occurrence' else f'{val:.2f}'
            color = 'white' if val > vmax * 0.6 else 'black'
            axes[0].text(j, i, text, ha='center', va='center', fontsize=annot_fontsize, color=color)

    im2 = axes[1].imshow(low_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels(cat_names, fontsize=tick_fontsize, rotation=30, ha='right')
    axes[1].set_title(f'Low Score (<{threshold_low})', fontsize=label_fontsize)

    for i in range(n_rows):
        for j in range(4):
            val = low_matrix[i, j]
            text = f'{val:.2f}' if value_type == 'occurrence' else f'{val:.2f}'
            color = 'white' if val > vmax * 0.6 else 'black'
            axes[1].text(j, i, text, ha='center', va='center', fontsize=annot_fontsize, color=color)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Occurrence Rate' if value_type == 'occurrence' else 'Share', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    plt.subplots_adjust(bottom=0.2, wspace=0.15)

    if fig_path:
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Saved to {fig_path}")

    return fig, {'high': high_matrix, 'low': low_matrix, 'stats': stats}


def filter_single_truck_factor_alarm(alarm_dict_raw: Dict[str, list]) -> Dict[str, list]:
    """
    Process alarm_dict_raw. If a repo's alarm list has only one element and it is 'truck_factor(year)',
    convert it to an empty list.

    Args:
        alarm_dict_raw: {repo_name: [alarm_list], ...}

    Returns:
        alarm_dict: processed new dict, original dict is not modified
    """
    import copy
    import re

    alarm_dict = copy.deepcopy(alarm_dict_raw)

    truck_factor_pattern = re.compile(r'^truck_factor\(.*\)$')

    for repo_name, alarm_list in alarm_dict.items():
        if isinstance(alarm_list, list) and len(alarm_list) == 1:
            if truck_factor_pattern.match(str(alarm_list[0])):
                alarm_dict[repo_name] = []

    return alarm_dict


def export_high_score_alarmed_repos(
    year_alarm_data: Dict[str, Dict],
    score_threshold: float = 7.0,
    csv_path: str = 'exp_result/0_rq/high_score_alarmed_repos.csv',
    json_path: str = 'exp_result/0_rq/high_score_alarmed_repos.json'
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Filter repos with scorecard score >= threshold and triggered alarms, export to CSV and JSON.

    Args:
        year_alarm_data: alarm data for a year, format: {repo_name: {'ossf_score': float, 'our_triggered': [...]}}
        score_threshold: scorecard score threshold, filter repos >= this score
        csv_path: output CSV file path (repo names only)
        json_path: output JSON file path (repo names and alarm reasons)

    Returns:
        (repo_list, alarm_dict):
            - repo_list: list of repo names meeting criteria
            - alarm_dict: {repo_name: [alarm_reasons]}
    """
    filtered_repos = []
    alarm_dict = {}

    for repo_name, data in year_alarm_data.items():
        ossf_score = data.get('ossf_score')
        triggered = data.get('our_triggered', [])

        # Filter condition: score >= threshold and has alarms
        if ossf_score is not None and ossf_score >= score_threshold and triggered:
            filtered_repos.append(repo_name)
            alarm_dict[repo_name] = triggered

    # Sort by repo name
    filtered_repos.sort()
    alarm_dict = {k: alarm_dict[k] for k in sorted(alarm_dict.keys())}

    # Export CSV (repo names only)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['repo_name'])
        for repo in filtered_repos:
            writer.writerow([repo])
    print(f"Saved CSV to {csv_path}, {len(filtered_repos)} repos total")

    # Export JSON (repo names and alarm reasons)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(alarm_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to {json_path}")

    return filtered_repos, alarm_dict