import pandas as pd
from scipy.stats.mstats import winsorize

def add_player_features(_group):
    """
        Description: This function analizes each player to obtain the variables:
            - date_first_wager
            - date_first_deposit
            - first_activity_month
            - activity_frequency
            - months_active_since_last_activity: time elapsed since the previous activity_month
        Input: this function's input is a grouped df, so it can be used as a groupby function.
    """
    # Find first deposit month if any
    first_dep = _group.loc[_group['total_deposit'] > 0, 'activity_month'].min()
    first_wager = _group.loc[_group['total_handle'] > 0, 'activity_month'].min()
    first_activity_month = _group['activity_month'].min()
    # Add as new columns
    _group['date_first_deposit'] = first_dep
    _group['date_first_wager'] = first_wager
    _group['first_activity_month'] = first_activity_month
    
    # Frequency: unique month counter, ordered
    _group = _group.sort_values('activity_month')
    _group['activity_frequency'] = (
        _group['activity_month'].rank(method='dense').astype(int)
    )
    _group['months_active_since_last_activity'] = (
        (_group['activity_month'].dt.to_period('M').astype('int') - _group['activity_month'].dt.to_period('M').astype('int').shift(1))
    )
    
    _group['months_active_since_last_activity'] = _group['months_active_since_last_activity'].fillna(0).astype(int)

    return _group



def winsorise_scipy(_df, _cols, _upper_lim=0.01, _lower_lim=0.0):
    for c in _cols:
        _df[c] = pd.Series(
            winsorize(_df[c], limits=(_lower_lim, _upper_lim)),  # returns masked array
            index=_df.index
        )
    return _df