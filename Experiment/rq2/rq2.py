import pickle
import os
import sys

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
exp_year_list = [2023, 2024, 2025]


if __name__ == '__main__':
    rq1_path = '../rq1/1_rq1.pkl'
    rq2_path = './2_rq2.pkl'

    with open(rq1_path, 'rb') as f:
        rq1_dict = pickle.load(f)


    if os.path.exists(rq2_path):
        with open(rq2_path, 'rb') as f:
            rq2_dict = pickle.load(f)
    else:
        rq2_dict = {}


    if 't1_1_occurrence_share' not in rq2_dict.keys():
        t1_1_result = rq2_t1_1_occurrence_and_share(
            rq1_dict['years_alarm'],
            metric_keys=METRIC.keys(),
            years=exp_year_list
        )
        rq2_dict['t1_1_occurrence_share'] = t1_1_result
        with open(rq2_path, 'wb') as f:
            pickle.dump(rq2_dict, f)
    else:
        t1_1_result = rq2_dict['t1_1_occurrence_share']


    if 't1_3_distribution_distance' not in rq2_dict.keys():
        t1_3_result = rq2_t1_3_distribution_distance(
            rq1_dict['years_alarm'],
            metric_keys=METRIC.keys(),
            years=exp_year_list
        )
        rq2_dict['t1_3_distribution_distance'] = t1_3_result
        with open(rq2_path, 'wb') as f:
            pickle.dump(rq2_dict, f)
    else:
        t1_3_result = rq2_dict['t1_3_distribution_distance']
    for pair, dist in t1_3_result['distances'].items():
        print(f"  {pair}: JSD={dist['jsd']:.4f}, TV={dist['tv']:.4f}")

    if 't1_4_permutation_jsd' not in rq2_dict.keys():
        t1_4_jsd_result = rq2_t1_4_permutation_test(
            rq1_dict['years_alarm'],
            metric_keys=METRIC.keys(),
            n_permutations=1000,
            distance_metric='jsd',
            random_seed=42
        )
        rq2_dict['t1_4_permutation_jsd'] = t1_4_jsd_result
        with open(rq2_path, 'wb') as f:
            pickle.dump(rq2_dict, f)
    else:
        t1_4_jsd_result = rq2_dict['t1_4_permutation_jsd']
    for pair, res in t1_4_jsd_result['results'].items():
        sig_marker = "*" if res['significant_0.05'] else ""
        print(f"  {pair}: d={res['observed_distance']:.4f}, p={res['p_value']:.4f} {sig_marker}")

    if 't1_4_permutation_tv' not in rq2_dict.keys():
        t1_4_tv_result = rq2_t1_4_permutation_test(
            rq1_dict['years_alarm'],
            metric_keys=METRIC.keys(),
            n_permutations=1000,
            distance_metric='tv',
            random_seed=42
        )
        rq2_dict['t1_4_permutation_tv'] = t1_4_tv_result
        with open(rq2_path, 'wb') as f:
            pickle.dump(rq2_dict, f)
    else:
        t1_4_tv_result = rq2_dict['t1_4_permutation_tv']
    for pair, res in t1_4_tv_result['results'].items():
        sig_marker = "*" if res['significant_0.05'] else ""
        print(f"  {pair}: d={res['observed_distance']:.4f}, p={res['p_value']:.4f} {sig_marker}")


    rq2_t1_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        value_type='occurrence',
        fig_path='./2_rq2_heatmap_12reasons_occurrence.pdf',
        cmap='Blues'
    )
    rq2_t1_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        value_type='share',
        fig_path='./2_rq2_heatmap_12reasons_share.pdf',
        cmap='Blues'
    )
    rq2_t1_category_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        value_type='occurrence',
        fig_path='./2_rq2_heatmap_4dimensions_occurrence.pdf',
        cmap='Blues'
    )
    rq2_t1_category_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        value_type='share',
        fig_path='./2_rq2_heatmap_4dimensions_share.pdf',
        cmap='Blues'
    )


    THRESHOLD_HIGH = 7.0
    THRESHOLD_LOW = 7.0


    if 't2_1_group_stats' not in rq2_dict.keys():
        t2_1_result = rq2_t2_group_stats(
            rq1_dict['years_alarm'],
            metric_keys=METRIC.keys(),
            years=exp_year_list,
            threshold_high=THRESHOLD_HIGH,
            threshold_low=THRESHOLD_LOW
        )
        rq2_dict['t2_1_group_stats'] = t2_1_result
        with open(rq2_path, 'wb') as f:
            pickle.dump(rq2_dict, f)
    else:
        t2_1_result = rq2_dict['t2_1_group_stats']


    cat_names = ['Project Meta Data', 'Test Module', 'Developer', 'Issue Management']
    print(f"  {'Year':<8} {'Group':<6} " + " ".join([f"{c:>12}" for c in cat_names]))
    for year in exp_year_list:
        for group in ['high', 'low']:
            shares = t2_1_result['by_year'][year][group]['category_occurrence']
            shares_pct = [f"{s*100:.2f}%" for s in shares]
            print(f"  {year:<8} {group.upper():<6} " + " ".join([f"{s:>12}" for s in shares_pct]))
    # Total
    for group in ['high', 'low']:
        shares = t2_1_result['total'][group]['category_occurrence']
        shares_pct = [f"{s*100:.2f}%" for s in shares]
        print(f"  {'Total':<8} {group.upper():<6} " + " ".join([f"{s:>12}" for s in shares_pct]))


    if 't2_2_group_difference' not in rq2_dict.keys():
        t2_2_result = rq2_t2_group_difference_test(
            rq1_dict['years_alarm'],
            metric_keys=METRIC.keys(),
            years=exp_year_list,
            threshold_high=THRESHOLD_HIGH,
            threshold_low=THRESHOLD_LOW,
            alpha=0.05
        )
        rq2_dict['t2_2_group_difference'] = t2_2_result
        with open(rq2_path, 'wb') as f:
            pickle.dump(rq2_dict, f)
    else:
        t2_2_result = rq2_dict['t2_2_group_difference']


    for reason, res in t2_2_result['total']['reasons'].items():
        sig = "*" if res.get('significant_fdr', False) else ""
        print(f"    {reason[:30]:30s}: RD={res['risk_difference']:+.3f}, OR={res['odds_ratio']:.2f}, p_fdr={res.get('p_fdr', res['p_value']):.4f} {sig}")


    for cat, res in t2_2_result['total']['categories'].items():
        sig = "*" if res.get('significant_fdr', False) else ""
        print(f"    {cat:20s}: RD={res['risk_difference']:+.3f}, OR={res['odds_ratio']:.2f}, p_fdr={res.get('p_fdr', res['p_value']):.4f} {sig}")


    rq2_t2_group_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        threshold_high=THRESHOLD_HIGH,
        threshold_low=THRESHOLD_LOW,
        value_type='occurrence',
        fig_path='./2_rq2_heatmap_highlow_occurrence.pdf',
        cmap='Blues'
    )
    rq2_t2_group_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        threshold_high=THRESHOLD_HIGH,
        threshold_low=THRESHOLD_LOW,
        value_type='share',
        fig_path='./2_rq2_heatmap_highlow_share.pdf',
        cmap='Blues'
    )

    rq2_t2_group_category_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        threshold_high=THRESHOLD_HIGH,
        threshold_low=THRESHOLD_LOW,
        value_type='occurrence',
        fig_path='./2_rq2_heatmap_highlow_4dim_occurrence.pdf',
        cmap='Blues'
    )
    rq2_t2_group_category_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        threshold_high=THRESHOLD_HIGH,
        threshold_low=THRESHOLD_LOW,
        value_type='share',
        fig_path='./2_rq2_heatmap_highlow_4dim_share.pdf',
        cmap='Blues'
    )



    if 't3_1_consistent_group_stats' not in rq2_dict.keys():
        t3_1_result = rq2_t3_consistent_group_stats(
            rq1_dict['years_alarm'],
            metric_keys=METRIC.keys(),
            years=exp_year_list,
            threshold_high=THRESHOLD_HIGH,
            threshold_low=THRESHOLD_LOW
        )
        rq2_dict['t3_1_consistent_group_stats'] = t3_1_result
        with open(rq2_path, 'wb') as f:
            pickle.dump(rq2_dict, f)
    else:
        t3_1_result = rq2_dict['t3_1_consistent_group_stats']



    cat_names = ['Project Meta Data', 'Test Module', 'Developer', 'Issue Management']
    print(f"  {'Year':<8} {'Group':<6} " + " ".join([f"{c:>12}" for c in cat_names]))
    for year in exp_year_list:
        for group in ['high', 'low']:
            occ = t3_1_result['by_year'][year][group]['category_occurrence']
            occ_pct = [f"{o*100:.2f}%" for o in occ]
            print(f"  {year:<8} {group.upper():<6} " + " ".join([f"{o:>12}" for o in occ_pct]))
    for group in ['high', 'low']:
        occ = t3_1_result['total'][group]['category_occurrence']
        occ_pct = [f"{o*100:.2f}%" for o in occ]
        print(f"  {'Total':<8} {group.upper():<6} " + " ".join([f"{o:>12}" for o in occ_pct]))

    if 't3_2_consistent_group_difference' not in rq2_dict.keys():
        t3_2_result = rq2_t3_consistent_group_difference_test(
            rq1_dict['years_alarm'],
            metric_keys=METRIC.keys(),
            years=exp_year_list,
            threshold_high=THRESHOLD_HIGH,
            threshold_low=THRESHOLD_LOW,
            alpha=0.05
        )
        rq2_dict['t3_2_consistent_group_difference'] = t3_2_result
        with open(rq2_path, 'wb') as f:
            pickle.dump(rq2_dict, f)
    else:
        t3_2_result = rq2_dict['t3_2_consistent_group_difference']


    for cat, res in t3_2_result['total']['categories'].items():
        sig = "*" if res.get('significant_fdr', False) else ""
        print(f"    {cat:20s}: RD={res['risk_difference']:+.3f}, OR={res['odds_ratio']:.2f}, p_fdr={res.get('p_fdr', res['p_value']):.4f} {sig}")


    rq2_t3_consistent_group_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        threshold_high=THRESHOLD_HIGH,
        threshold_low=THRESHOLD_LOW,
        value_type='occurrence',
        fig_path='./2_rq2_heatmap_consistent_occurrence.pdf',
        cmap='Blues'
    )
    rq2_t3_consistent_group_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        threshold_high=THRESHOLD_HIGH,
        threshold_low=THRESHOLD_LOW,
        value_type='share',
        fig_path='./2_rq2_heatmap_consistent_share.pdf',
        cmap='Blues'
    )

    rq2_t3_consistent_group_category_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        threshold_high=THRESHOLD_HIGH,
        threshold_low=THRESHOLD_LOW,
        value_type='occurrence',
        fig_path='./2_rq2_heatmap_consistent_4dim_occurrence.pdf',
        cmap='Blues'
    )
    rq2_t3_consistent_group_category_heatmap(
        rq1_dict['years_alarm'],
        metric_keys=METRIC.keys(),
        years=exp_year_list,
        threshold_high=THRESHOLD_HIGH,
        threshold_low=THRESHOLD_LOW,
        value_type='share',
        fig_path='./2_rq2_heatmap_consistent_4dim_share.pdf',
        cmap='Blues'
    )

