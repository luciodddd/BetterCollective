import os
import pandas as pd
from pathlib import Path
import openpyxl
import numpy as np
import Functions #I created the functions in a separate ".py" so this code looks cleaner.
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

""" NOTE: There are many prints as I used them to visualize some results, fell free to comment them for a cleaner result on console."""

#Path actual:
_current_dir = Path(__file__).parent
_dataset_path = _current_dir.parent / "Files/sample_data__technical_assessment_1.xlsx"


# ----------------------------  load + quick schema peek ----------------------------
_df = pd.read_excel(_dataset_path)
#Check Schema, most resent activity month and general information:
print("Data types: ",_df.dtypes)
print("First look: ", _df.describe()[[ "total_deposit", "total_handle", "total_ngr"]]) #There are no negative deposits and no negative bettings. There are negative
# revenue but that's possible, it would mean the firm has negative revenue (a loss) with those users.
print("Most resent activity month: ",_df["activity_month"].max()) #I assume the exercise is as if we were located in this period of time, as this is the most recent data.
print(f"Unique players: {_df['account_id'].nunique()}")
print("Activity date range: \nFrom: ", _df['activity_month'].min(), "To: ", _df['activity_month'].max())
# First I delete duplicated entries at a general level (consulting all columns), but there are no duplicated entries on the data.
print("Rows: ",_df["account_id"].count())
_df = _df.drop_duplicates() 
print("Rows after deleting global duplications: ",_df["account_id"].count())

"""I check for duplicated data in the subset "activity_month" and "account_id". I see there are multiple entries for some account-activity_month on different brands. 
The exercise announcemente states there should be only one entry per player+month, so there are 3 options. If I don't want to use brand as a model variable and instead
focus on the player I could group the info per activity_month and account_id. If I wanna keep brand_id, then I should erase all entries with this issue, since erasing
on of them is biased. I'll remove the players with the duplicated entries for now."""

_duplicated_month_player = _df[_df.duplicated(subset=["activity_month","account_id"])]
print(_duplicated_month_player)
_df = _df[~_df["account_id"].isin(_duplicated_month_player["account_id"].unique().tolist())]
# There are NaN values at "total_deposit" and "total_handle", so I will fill them up with 0s. This will distort the analysis at some extent but I plan on filtering false activity
# later on:
print("NaNs deposits: ", _df[_df["total_deposit"].isna()]["total_deposit"].count())
print("NaNs wager: ", _df[_df["total_handle"].isna()]["total_handle"].count())
_df[["total_deposit","total_handle"]] = _df[["total_deposit","total_handle"]].fillna(0)
# If we consider a sing up an activity, then I'll check cases where activity_month("%Y-%m")!=reg_date("%Y-%m") and total_deposit==0 and total_handle==0. Since there is no real activity it shouldn't be considered a "activity_month":
_df['fake_activity_flag'] = (
    (_df['total_deposit'] == 0) &
    (_df['total_handle'] == 0) 
)
#After finding out those players didn't have any posterior activity, I removed this condition previously used: &(_df['activity_month'].dt.to_period('M') != _df['reg_date'].dt.to_period('M'))
_df = _df[~_df['fake_activity_flag']].drop(["fake_activity_flag"],axis=1)

# I check if the data is historic or cumulative. This means if a "activity_month"+"account_id" is the latest month of activity and acumulates all deposits, bets and revenue for the firm, or if there is an entry for each month the player has done something (a deposit or a bet).
_activity_months_by_account = (
    _df.groupby("account_id")["activity_month"]
    .nunique()
    .reset_index(name="activity_frecuency")
)
_activity_frecuency = (
    _activity_months_by_account["activity_frecuency"]
    .value_counts()
    .sort_index()
    .reset_index()
    .rename(columns={"index": "num_unique_months", "activity_frecuency": "account_count"})
)
_activity_frecuency["% Percentage"] = _activity_frecuency["count"]/_df["account_id"].nunique()
print("Historic or Cumulative data?: ",_activity_frecuency)
""" 88% of accounts have 1 activity month and 12% have 2 or more. As the 12% of multiple-month entries are real data (are not duplicated or empty)
I'll assume it's historic data, and each entry of activity_month is when the player made a deposit or bet (activity). Now, if this is true,
this variable of "activity_frecuency" could be more important than "activity_month", which only measures the time elapsed between the
last time the player did someting and the first deposit's date.
So I will add columns, some of which I will use in the next checks:
    1) activity_frecuency: Counts how many times the player was active at that moment.
    2) date_first_deposit: stores the date of the first deposit in this dataset if the player ever had the column "total_deposit" become >0. So later on I can check for NaT ftd_date and try to fix it.
    3) date_first_wager: stores the date of the first wage in this dataset if the player ever had the column "total_handle" become >0. So later on I can check for NaT qp_date and try to fix it.
    4) first_activity: stores the first "activity_month" the user had, to later check if reg_date < first_activity_month and first_activity_month <= ftd_date.
    5) months_active_since_last_activity: stores the time elapsed between the previous "activity_month" and the current one. If there is no other activity_month then it is equal to "months_active" (which will be generated after the cleaning)
"""
_df = _df.sort_values(by=['account_id', 'activity_month'])
_df = _df.groupby('account_id').apply(Functions.add_player_features, include_groups=False).reset_index(level=0)
print(_df)
# If first_activity_month < reg_date that is a wrong data that should not be considered:
_df = _df[_df["first_activity_month"].dt.to_period('M')>=_df["reg_date"].dt.to_period('M')]

# A missing value on "ftd_date" and "qp_date" should mean the user didn't make a deposit/bet yet. So I'll check for:
    # If a player has "ftd_date"==NaT and total_deposit>0 for any month with activity there are 2 options: 
    # 1) The data is wrong and "ftd_date" should have had a value when the deposit became positive for the first time ("date_first_deposit" calculated before)
    # 2) It doesn't have any deposit, and it doesn't have any wage either, and the activity_month is the same period than the register period.
    # or 3) there was a promotion, event, free money trial, etc. that counts as deposit and allows the user to bet without making a deposit. But as the column is defined as "how much money the player deposited" I will consider case 1.
print("There are 5885 entries with ftd_date as .NaT: ",_df[_df["ftd_date"].isna()]) # This means we can't calculate "months_active" as the exercise suggests
#I clean miss-inputs using the columns I created beforehand. I am taking the first activity_month with deposit>0 or wager>0, which in case of no missing data, should be equivalent.
_df["ftd_date"] = np.where(_df["ftd_date"].isna(),_df["date_first_deposit"],_df["ftd_date"])
_df["qp_date"] = np.where(_df["qp_date"].isna(),_df["date_first_wager"],_df["qp_date"])


""" There is another issue with the data. For some players, ftd_date or qp_date exists and is previous to the minimum activity_month for that player, which means there is missing data that can't be reproduced.
All dates have to be equal or greater to the sing up date ("activity_month","ftd_date","qp_date") but that's not the case, there are entries where ftd_date is lower than reg_date.
In this exersise I'll leave like this, but I would ask the IT department how those variables are calculated to check which column should be corrected"""


# Add the months_active columns. I defined this as the difference between activity_month and the minimum between ftd_date or qp_date, as both are activities:
_df["first_activity"] = _df[["ftd_date", "qp_date"]].min(axis=1)
_df["months_active"] = (
    (_df["activity_month"].dt.year - _df["first_activity"].dt.year) * 12
    + (_df["activity_month"].dt.month - _df["first_activity"].dt.month)
)
# I fill 0s of "months_active_since_last_activity" with "months_active", as I said when I created the column:
_df["months_active_since_last_activity"] = _df["months_active_since_last_activity"].fillna(_df["months_active"])
# I create a variable "average_time_to_activity" to investigate how often a player plays:
_df["average_time_to_activity"] = _df["months_active"] / _df["activity_frequency"] #Time from the first activity divided by the number of months the player was active

print("Average time to activity: ", _df["months_active"].sum()/ _df["activity_frequency"].sum()) # It's 3.6 months

print("\nmonths_active stats:")         
print(_df['months_active'].describe())

# Deposits vs Handle correlation
corr = _df[['total_deposit', 'total_handle', 'total_ngr']].corr()
print("\n Correlations between deposit, handle, NGR:\n", corr) 
""" 
    * Strong positive correlation between deposits and wager, expected and good to see.
    * Almost zero total_handle-total_ngr correlation, the wager doesn't predict much of sportsbook firm net revenue. This could be because of the sample, wagers tend to win a lot. Or because most wagers use promotion coins or freebies to gamble.
    * Positive but low tota_deposit-total_ngr correlation, suggests that how much a player deposits doesn't strongly predict how profitable they are for the operator either.
    * There's high variance in player profitability.
"""

# Players with no deposit but wagers:
_no_deposit_but_wager = _df.copy()
_no_deposit_but_wager = _no_deposit_but_wager.groupby(by=['account_id']).agg({"total_deposit":"sum","total_handle":"sum"}).reset_index()
_no_deposit_but_wager = _no_deposit_but_wager[(_no_deposit_but_wager['total_deposit'] == 0) & (_no_deposit_but_wager['total_handle'] > 0)]
print(f"\nPlayers who wagered but never deposited: {_no_deposit_but_wager['account_id'].nunique()}")

# ------- GRAPHICS, some extra exploration:

print("Activity per player:")
plt.figure(figsize=(8,4))
_activity_months_by_account.hist(bins=30)
plt.title('Distribution: Activities Per Player')
plt.xlabel('Amount of Months Active')
plt.ylabel('Number of Players')
plt.show()

# ----> Quick log-hist to see long tails, I used log to avoid any distortion that whales (real big wagers) could have, besides its other benefits
_num_cols = ['total_deposit', 'total_handle', 'total_ngr','months_active', 'activity_frequency','average_time_to_activity']
# On positive columns only:
for c in ['total_deposit', 'total_handle']:
    _df[c + '_log'] = np.log1p(_df[c])
# signed-log for total_ngr because it has negative values:
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))
_df['total_ngr_log'] = signed_log1p(_df['total_ngr'])
# drop any ±inf that might sneak in (like a total_ngr == –1)
_df = _df.replace([np.inf, -np.inf], np.nan)
_df[['total_deposit_log', 'total_handle_log', 'total_ngr_log','months_active', 'activity_frequency', 'average_time_to_activity']].hist(
        bins=50, figsize=(15, 10)
)
plt.tight_layout()
plt.show()
""" NOTE: Results are shown in Log-Hist-Analysis.png
    * Deposits: ~ 75 % of rows are “no-deposit” months. Long but very thin right tail up to ≈ log 13 (aprox. $440 k, most likely whales)
    * Wagers: Shows a centred body with fat tails on both sides. Similar to deposits, it has a high bar at value 0 for small bets.
    * I could add some flag columns to check has_deposit / has_handle. I could also log the scales so whales don't distort a loss function (if I use one later).
    * Net Revenue: Kind of simetrical around 0. I could add a column "is_profitable_month" and a "abs_ngr_log" if I needed the absolute value.
    * Months Active:Tall bar at 0, then rapid drop. Tiny hump between 5-15 months. This could mean a lot of players churned.
    * Activity Frecuency: Most players were only active for 1 month. Only 3% reach 3+ different months of activities
    * average_time_to_activity: Monstly mirror "Months Active" since they played for only 1 month. A few high values when players return after long gaps.

    * Decitions: I will winsorise total_deposit and total_handle at the 99th percentile (<1%).
"""
_cols_to_cap = ["total_deposit", "total_handle"]
_df = Functions.winsorise_scipy(_df, _cols_to_cap, _upper_lim=0.01, _lower_lim=0.0)
# I log it
for c in _cols_to_cap:
    _q99 = _df[c].quantile(0.99)
    print(f"{c} 99th-pct cap: {_q99:,.0f}")

# It also showed me there are 131 cases where _df['months_active']<0. This is due to a activity_month lower than the minumum between ftd_date and qp_date.
# I could set the months_active to 0 and keep the data, but as there are few cases I'll remove them, as there might be a error in the data input (If there is an activity in that month it has to be a deposit or a bet, so it should not be possible for ftd_date and pq_date to be higher)
_df = _df[_df['months_active']>=0]

# --> Retention curve:
kmf = KaplanMeierFitter()
# churn = first month where player disappears for ≥3 months (simple rule for checking)
_last_month = _df.groupby('account_id')['activity_month'].max().dt.to_period('M')
_event_time = _df.groupby('account_id')['months_active'].max()
_churned = _last_month < _df['activity_month'].max().to_period('M') - 2  # ≥3 mo gap

kmf.fit(_event_time, event_observed=_churned, label='Overall')
kmf.plot_survival_function()
plt.title('Player Retention Curve')
plt.show()

""" NOTE: Results are in "Player Retention Curve - EDA.png"
    * This is a graph where: x-axis: month_active. y-axis: survival probability (fraction still active). Steps are each time at least 1 player chruns during that time.
    * 40% of players churn after their first active month.
    * More than half of the remaining cohort is gone within 3 months
    * Only 1 in 4 players is still active six months in.
    * A typical player churns 3 to 4 months after first activity.

"""

# I will clean the columns I'll not use and store the dataframe as a parquet to use it later to train the model. (I could also save it in a sql server, s3, etc.)
# Drop helper columns:
_df = _df.drop(["date_first_deposit","date_first_wager","first_activity_month","first_activity","months_active_since_last_activity"],axis=1)
_clean_data_path = _current_dir.parent / "data/cleaned_data_for_fcst.parquet"
_df.to_parquet(_clean_data_path, engine="pyarrow", compression="snappy", index=False)
print("Wrote:", _clean_data_path)
print("File exists?", _clean_data_path.exists()) 