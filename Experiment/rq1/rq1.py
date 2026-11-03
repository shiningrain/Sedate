import pickle
import json
from datetime import date
import os
import sys
from typing import Dict, Any, Tuple, Optional, List
import math
import numpy as np
import matplotlib.pyplot as plt

# Add exp_utils path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Code', 'utils'))
from exp_utils import *

METRIC = {
    'code_churn_rate(%, 3week)':'tau_cc',
    'burstiness(year)':'tau_bst',
    'commit_frequency(year)':'tau_cf',
    'test_proportion(%, year)':'tau_tp',
    'test_code_churn_ratio(3week)':'tau_ter',
    'core_developer_turnover(year)':'tau_cdt',
    'truck_factor(year)':'tau_tf',
    'newcomer_retention(%, year)':'tau_nr',
    'issue_response_efficiency(day, year)':'tau_ire',
    'backlog_management_index(%, year)':'tau_bmi',
    'cve_exposure_increase(year)':'tau_cer',
    'cve_severity(year)':'tau_cs'
}
exp_year_list=[2023, 2024, 2025]


def get_our_metric_by_year(our_metric):

    years = exp_year_list
    metric_new = {}

    for repo_name, value_dict in our_metric.items():
        year_list = value_dict.get('years', [])
        year_to_idx = {y: i for i, y in enumerate(year_list) if y in years}

        repo_metrics = {}
        for metric_key in METRIC.keys():
            metric_values = value_dict.get(metric_key, [])
            repo_metrics[metric_key] = {
                y: metric_values[idx] if idx < len(metric_values) else None
                for y, idx in year_to_idx.items()
            }
            for y in years:
                if y not in repo_metrics[metric_key]:
                    repo_metrics[metric_key][y] = None

        metric_new[repo_name] = repo_metrics

    return metric_new


def alarm_threshold_from_new_structure(our_metric_by_year, year):

    alarm_dict = {}
    for repo_name, year_data in our_metric_by_year.items():
        triggered_metrics = year_data.get(year, {})
        alarm_dict[repo_name] = list(triggered_metrics.keys())
    return alarm_dict


def alarm_threshold(our_metric_by_year, year, config_path='test_code/alarm_config.json'):

    with open(config_path, 'r') as f:
        config = json.load(f)

    tau_to_metric = {v: k for k, v in METRIC.items()}

    threshold_checks = []
    for thresh_config in config['thresholds'].values():
        for key, val in thresh_config.items():
            if key.startswith('tau_') and key in tau_to_metric:
                threshold_checks.append({
                    'metric_key': tau_to_metric[key],
                    'threshold': val,
                    'direction': thresh_config['direction']
                })

    alarm_dict = {}
    for repo_name, metrics in our_metric_by_year.items():
        triggered = []
        for check in threshold_checks:
            val = metrics.get(check['metric_key'], {}).get(year)
            if val is None:
                continue
            if (check['direction'] == 'g' and val >= check['threshold']) or \
               (check['direction'] == 'l' and val <= check['threshold']):
                triggered.append(check['metric_key'])
        alarm_dict[repo_name] = triggered

    return alarm_dict


def combine_ossf_alarm(ossf_by_year, alarm_dict, year):
    result = {}
    all_repos = set(ossf_by_year.keys()) | set(alarm_dict.keys())
    for repo_name in all_repos:
        result[repo_name] = {
            'ossf_score': ossf_by_year.get(repo_name, {}).get(year),
            'our_triggered': alarm_dict.get(repo_name, [])
        }
    return result


def get_ossf_by_checkpoint(ossf_result, checkpoint_month=9, checkpoint_day=2):

    years = exp_year_list#
    ossf_new = {}

    for repo_name, value_dict in ossf_result.items():
        # if repo_name!='github.com/aws/karpenter-provider-aws':
        #     continue
        repo_scores = {}
        for year in years:
            checkpoint = date(year, checkpoint_month, checkpoint_day)
            best_date, best_score = None, None

            for key, val in value_dict.items():
                if not key.startswith('d_') or not isinstance(val, (int, float)) or val < 0:
                    continue
                parts = key.split('_')
                if len(parts) != 4:
                    continue
                try:
                    key_date = date(int(parts[1]), int(parts[2]), int(parts[3]))
                except ValueError:
                    continue
                if key_date <= checkpoint and (best_date is None or key_date > best_date):
                    best_date, best_score = key_date, val

            repo_scores[year] = best_score
        ossf_new[repo_name] = repo_scores

    return ossf_new


if __name__ == '__main__':
    rq1_path = './1_rq1.pkl'
    rq_save_dir='.'
    bucket_template_path='./1_rq1_bucket-{}.pdf'
    combined_fig_path = './1_rq1_bucket-combined.pdf'
    lines_fig_path = './1_rq1_alert_rate_lines.pdf'

    if not os.path.exists(rq_save_dir):
        os.makedirs(rq_save_dir)

    if os.path.exists(rq1_path):
        with open(rq1_path, 'rb') as f:
            rq1_dict = pickle.load(f)
    else:
        rq1_dict={}


    ossf_by_year=rq1_dict['raw_results_by_year']['ossf']
    our_metric_by_year=rq1_dict['raw_results_by_year']['our']

    none_result=[]
    for repo_name, year_scores in ossf_by_year.items():
        for year in exp_year_list:
            if year_scores.get(year) is None:
                none_result.append((repo_name, year))
    print('None result cases: ',len(none_result))


    if 'years_alarm' not in rq1_dict.keys():
        year_dict={}
        for year in exp_year_list:
            alarm_dict = alarm_threshold_from_new_structure(our_metric_by_year, year)
            year_dict[year]=combine_ossf_alarm(ossf_by_year, alarm_dict, year)
        rq1_dict['years_alarm']=year_dict
        with open(rq1_path, 'wb') as f:
            pickle.dump(rq1_dict, f)
    else:
        year_dict=rq1_dict['years_alarm']

    if 'hypothesis_total' not in rq1_dict.keys():
        bucket_dict={}
        for year in exp_year_list:
            scores_np, alarms_np, bucket_stats = rq1_plot_score_alarm_buckets(
                rq1_dict['years_alarm'][year], bin_width=1.0, fig_path=bucket_template_path.format(year))
            bucket_dict[year]=bucket_stats
        
        combined_bucket = []
        for i in range(len(bucket_dict[2023])):
            n_total = sum(bucket_dict[year][i]['n_total'] for year in exp_year_list)
            k_alarm = sum(bucket_dict[year][i]['k_alarm'] for year in exp_year_list)
            alarm_rate = (k_alarm / n_total * 100) if n_total > 0 else 0
            combined_bucket.append({
                'bucket': bucket_dict[2023][i]['bucket'],
                'n_total': n_total,
                'k_alarm': k_alarm,
                'alarm_rate': alarm_rate
            })


        low_score_n_total = sum(b['n_total'] for b in combined_bucket[:3])
        low_score_k_alarm = sum(b['k_alarm'] for b in combined_bucket[:3])
        low_score_alarm_rate = (low_score_k_alarm / low_score_n_total * 100) if low_score_n_total > 0 else 0
        print(f"Score 0-3: n_total={low_score_n_total}, k_alarm={low_score_k_alarm}, alarm_rate={low_score_alarm_rate:.2f}%")


        low_score_n_total = sum(b['n_total'] for b in combined_bucket[-3:])
        low_score_k_alarm = sum(b['k_alarm'] for b in combined_bucket[-3:])
        low_score_alarm_rate = (low_score_k_alarm / low_score_n_total * 100) if low_score_n_total > 0 else 0
        print(f"Score 7-10: n_total={low_score_n_total}, k_alarm={low_score_k_alarm}, alarm_rate={low_score_alarm_rate:.2f}%")


        _, _, total_bucket_stats = rq1_plot_score_alarm_buckets(combined_bucket, fig_path=combined_fig_path)


        rq1_plot_alarm_rate_lines(bucket_dict, combined_bucket, fig_path=lines_fig_path)

        rq1_dict['visual_bucket_by_year']=bucket_dict
        rq1_dict['visual_bucket_total']=total_bucket_stats


        hypothesis_by_year = {}
        for year in exp_year_list:
            hypothesis_by_year[year] = rq1_logistic_alarm_hypothesis_test(
                rq1_dict['years_alarm'][year])
        rq1_dict['hypothesis_by_year'] = hypothesis_by_year

        combined_repo_dict = {}
        for year in exp_year_list:
            for repo_name, data in rq1_dict['years_alarm'][year].items():
                key = f"{repo_name}_{year}"
                combined_repo_dict[key] = data
        rq1_dict['hypothesis_total'] = rq1_logistic_alarm_hypothesis_test(combined_repo_dict)

        with open(rq1_path, 'wb') as f:
            pickle.dump(rq1_dict, f)
    else:
        bucket_dict=rq1_dict['visual_bucket_by_year']
        total_bucket_stats=rq1_dict['visual_bucket_total']
