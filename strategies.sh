## FOR PASCAL-0-1 top-5 best strategies
strategies = [
    # Best top-n values is 0.9, for percentile is 0.1
    {'type': 'top_n', 'value': 0.9, 'group_by_class': False, 'name': 'top_n_overall', 'creative': False},
    {'type': 'percentile', 'value': 0.1, 'group_by_class': False, 'name': 'percentile_overall', 'creative': False},
    {'type': 'percentile', 'value': 0.1, 'group_by_class': False, 'name': 'percentile_overall_creative', 'creative': True},
    {'type': 'top_n', 'value': 0.9, 'group_by_class': False, 'name': 'top_n_overall_creative', 'creative': True},
    {'type': 'percentile', 'value': 0.1, 'group_by_class': True, 'name': 'percentile_class_creative', 'creative': True},
]

# FOR PASCAL-0-2 top-5 best strategies
    # Best top-n values is 0.9 in mixed, 0.8 in solo, for percentile is 0.1 in mixed, z_score in mixed best is 2
strategies = [
    {'type': 'percentile_top_n', 'percentile_value': 0.1, 'top_n_value': 0.9, 'group_by_class': False, 'name': 'percentile_top_n_overall', 'creative': False},
    {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.8, 'group_by_class': False, 'name': 'zscore_top_n_overall', 'creative': False},
    {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.9, 'group_by_class': False, 'name': 'zscore_top_n_overall_creative', 'creative': True},
    {'type': 'top_n', 'value': 0.8, 'group_by_class': False, 'name': 'top_n_overall_creative', 'creative': True},
    {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.9, 'group_by_class': False, 'name': 'zscore_top_n_overall', 'creative': False},
]

# FOR PASCAL-0-4 top-5 best strategies
    # Best top-n values is 0.8/0.7 in mixed, 0.6 in solo, for percentile is 0.1 in mixed, z_score in mixed best is 2
strategies = [
    {'type': 'percentile_top_n', 'percentile_value': 0.1, 'top_n_value': 0.8, 'group_by_class': False, 'name': 'percentile_top_n_overall_creative', 'creative': True},
    {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.7, 'group_by_class': True, 'name': 'zscore_top_n_class_creative', 'creative': True},
    {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.7, 'group_by_class': True, 'name': 'zscore_top_n_class', 'creative': False},
    {'type': 'top_n', 'value': 0.6, 'group_by_class': False, 'name': 'top_n_overall_creative', 'creative': True},
    {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.8, 'group_by_class': True, 'name': 'zscore_top_n_class', 'creative': False},
]

# Top 5 best strategies for PASCAL-0-8 dataset
# Best are combined strategies, z_score is good with 1.5/2 values, top_n from 0.7 to 0.9, percentile value is good at 0.1
strategies = [
    {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.9, 'group_by_class': False, 'name': 'zscore_top_n_overall_creative', 'creative': True},
    {'type': 'zscore', 'threshold': 1.5, 'group_by_class': False, 'name': 'zscore_overall'},
    {'type': 'percentile_top_n', 'percentile_value': 0.1, 'top_n_value': 0.9, 'group_by_class': False, 'name': 'percentile_top_n_overall', 'creative': False},
    {'type': 'zscore_top_n', 'zscore_threshold': 1.5, 'top_n_value': 0.8, 'group_by_class': True, 'name': 'zscore_top_n_class_creative', 'creative': True},
    {'type': 'zscore_top_n', 'zscore_threshold': 1.5, 'top_n_value': 0.7, 'group_by_class': False, 'name': 'zscore_top_n_overall_creative', 'creative': True},
]

# Top 5 best strategies fro PASCAL-0-16 dataset:
strategies = [
    {'type': 'percentile_by_columns', 'value': 0.1, 'group_by_class': False, 'name': 'percentile_columns_overall'},
    {'type': 'percentile', 'value': 0.2, 'group_by_class': False, 'name': 'percentile_overall', 'creative': False},
    {'type': 'percentile_top_n', 'percentile_value': 0.1, 'top_n_value': 0.9, 'group_by_class': True, 'name': 'percentile_top_n_class', 'creative': False},
    {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.8, 'group_by_class': False, 'name': 'zscore_top_n_overall_creative', 'creative': True},
    {'type': 'percentile_by_columns', 'value': 0.2, 'group_by_class': False, 'name': 'percentile_columns_overall'},
]