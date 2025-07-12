-------------- AT THE END OF THIS FILE IS THE GITHUB STRUCTURE AND WHERE TO FIND THINGS.
1.1.1) and 1.1.2) Provide the code you use to explore the data including any notes on statistics or trends that you find interesting.
Are there any obvious issues with the data? What challenges do you foresee based on this initial exploration?

---> Code on "Code/Data Exploration.py" Here I have comments with my thought process
---> Some graphics on "Graphics/..."
Some of the main treatments I found:
* Duplicate player-month rows: a few players appear twice in the same month (different brand IDs). Fix: I dropped every player that had this conflict so we keep the 'one row per player-month' rule.
* Fake activity rows: Some records have zero deposit and zero handle but are logged as a separate month right after registration. Risk: inflates activity counts and mis-labels churn. Fix: removed those rows.
* Missing first-deposit or first-bet dates: ~5800 rows had ftd_date or qp_date as NaT even though money columns were positive. Risk: impossible to compute months_active. Fix: imputed the first month where deposit or handle > 0 for that player.
* Negative months_active: 131 rows where activity_month precedes the imputed first-activity date. Risk: breaks survival math. Fix: dropped them because there were not many cases ( <1 % of data).
* Extreme right-skew ('whales'): top 1 % deposits > $5 k, handle > $30 k. Risk: a handful of whales dominate model loss. Fix: winsorised those two columns at the 99-th percentile and used log-scales.

 -- Assumptions made:
* Row meaning: each record is a single player-month of real monetary activity. If total_deposit = total_handle = 0 and it’s just a registration echo, I flagged it as “fake activity.”
* Missing monetary values: NaN in total_deposit or total_handle means no money moved. I replaced with zero.
* Missing first-deposit / first-bet dates: If ftd_date or qp_date is missing while money is > 0, I assumed the true date equals the player’s first positive month and impute it.
* Churn definition: a player is labelled 'churned' after three consecutive silent months. Quick-churn = lifetime ≤ 3 months and the player eventually meets the churn rule.
* Some parameters and thresholds when modeling.

2.1) Implement a baseline model that predicts churn for months 0 to 60. You can choose any algorithm you find appropriate . Provide the code used to train and evaluate the model as well as brief notes on why you chose the model you did. The goal is to see how players churn out over time

--> Answered at "Code/Baseline training.py"

2.2) Build a model that classifies or predicts the likelihood of whether or not a customer (or segment of customers) will churn. Provide the code used to train and evaluate the model as well as brief notes on why you chose the model you did. The goal is to predict early on the likelihood a player will churn out quickly and therefore have a low CLV

--> Answered at "Code/Early Chrun Classifier.py"

3.1.1) How would you evaluate the effectiveness of this model over time as new data continues to come in?

-->Code in "monitoring_pipeline.py"
First, I'd watch the core ML metrics exactly the same way I measured them at training time. Every night the batch-scoring job would write each player’s probability to a 'predictions-table'. As soon as a player's churn label becomes known (three silent months have passed), I'd back-fill that row with the ground-truth flag. Once a week I'd roll a 60-day window over those fully-labelled rows and recompute AUC-ROC, AUC-PR, and Brier score, to answer "When we predicted high churn risk, did those players actually leave?". Using the same metrics guarantees an apples-to-apples comparison and if AUC-ROC ever slips more than 5% points below the training benchmark it means the model is getting dull and needs retraining.
Second, I'd verify calibration. Good rank-ordering is not enough when the scores will feed a CLV pipeline—probabilities must be well-scaled. I plot a reliability curve ("expected vs. observed" chart) and compute Expected Calibration Error (ECE). If ECE drifts above 0.05 I'd run a quick recalibration step (a small correction layer) without touching the trees themselves.
Third, as there is going to be 2000 combination partner-state, and sometimes only a few of them have problems or change their patterns, I'd look for segment-level drift rather than retraining the entire fleet blindly. For each player and state I calculate a population-stability index (PSI) on key features such as log-deposit (compare today's input data to the data I trained on). When PSI crosses the usual 0.2 threshold I retrain only that partner's model, everyone else keeps their stable version. This keeps compute costs and MLOps noise down when we're talking about roughly 2 000 parallel models.
Finally, I account for label latency. Because “quick churn” is only confirmed after three months of silence, the evaluation pipeline is designed to back-fill labels nightly. . That way my weekly metrics always use the most complete and up-to-date truth.
In short, the model'ss health would be checked weekly for performance, calibration, and drift, with clear thresholds that either trigger an automatic recalibration or a full retrain. That'd give us continuous proof the model is still adding value, not just the day we shipped it.

3.1.2) What alternatives would you consider if you have more data and time? 

If I had more data and time I would try multiple models to check which fits better the information. Mdels like Survival Gradient-Boosting, that combines the "who" and "when" spects into a single hazard curve, giving us direct CLV estimates instead of a binary quick-churn flag. I would also research for better models that have already worked well for similar application cases.
I would use the whole player timeline, not just month 0. Entire monthly sequence of deposits, handle, wins/losses, promo usage (if there was any), so we could capture the trajectory patterns like 'early hot streak followed by cooldown' vs. 'steady small bets'. And we could add that behaviour to the models.
If there are 2k combinations of partner+geographies, I could also explore the option for the model to learn across partners, we could cluster partners with identical behaviour to help those with just a few entries to train the model.
I would also add checks to features (control which features and in which format they are served into the model), add conections to ddbb, etc.

3.2.1) How would you approach the fact that this model needs to be repeated approximately 2000 times for different partners and geographies combinations?

I would follow this steps:
1- I'd have all logic (feature engineering, preprocessing, XGBoost hyper-params, evaluation, model-card generation) in one file train.py, an opp class, or Docker Image, with the layout of the training and no hard-codings.
2- The only thing that would change among series (partner-state) would be a JSON or YAML with the parameters for each individual partner-state, so later I can use Dagster to read a master table of such configurations and instantiate 2k parallel tasks. So it would be one operator, many configurations.
3- For the parallel execution I'd try to use elastic computation, like Kubernetes Jobs or AWS Batch, auto-scaling. 
4- Each task mounts the same Docker image, pulls its YAML/JSON, trains, evaluates, writes artefacts, etc.
5- I'd store the results in versioned partitions. For example, if it was exported so S3: s3://bc-churn-models/{partner_id}/v{run_id}/model. A MLflow Tracking with a Postgres backend with tags like partner_id, state, data_start, data_end.
6- As there's going to be many series, I would maybe add a selective retraining, like I stated before. Nightly monitoring (PSI + AUC drift) writes a needs_retrain flag into a control table.

3.2.2) Where would you save the outputs of the models? In which format? Providing that is going to be a batch run.

For a nightly batch-scoring run I would write each player’s churn probability to an immutable, column-oriented file. I would write it in a cloud object-storage. For example, if it were AWS S3 bucket, I would lay it out like this:
s3://bc-churn-scores/
       date=2025-07-13/     # daily partition
           partner=AZ01/    # sub-partition, partnet
               scores.snappy.parquet
           partner=CO02/    # sub-partition, partnet
               scores.snappy.parquet
       date=2025-07-14/	    # daily partition
           … # and so on.
I would save them on Parquet + Snappy compression, so I can scan just the churn_prob or a single partner without reading the whole file. I could read it on Athena/Presto/Spark/Polars that are fast.
I selected storage instead of a database because it's going to be million of rows per night and S3 is cheap, scales transparently, and can be easily integrated with every downstream analytics or BI tool.
Marketing or CRM teams that need yesterday's scores can query the partition directly or load a slice into Redshift/Snowflake, no ETL bottleneck.
If a downstream system needs a lightweight table (like 'top-10% at-risk players per partner'), a daily Spark job can read the Parquet partition, aggregate, and write a small CSV or Postgres table.
I would probably set a retention policy, like to keep 18 months of daily partitions in S3 Standard, then migrate to Glacier to satisfy audit requirements without wasting hot-storage costs.

3.2.3) Once you have your training code, prepare an additional .py that uses principles of OOP to deliver the model as an artifact

--> Answer on code "churn_model_oop.py"

Objects can be created as:
(python)
from churn_model_oop import EarlyChurnModel
model = EarlyChurnModel.load("artifacts/AZ01")
scores = model.predict_proba(new_player_dataframe)    # <--- This one is the alternative constructor.
(you could also run it in a shell like bash)
python churn_model_oop.py \
       --input  Files/cleaned_AZ01.parquet \          # <--- This 2 are the required arguments of the main function, and are the new data's df path and artifact's output path
       --output artifacts/AZ01


Repository structure:
BetterCollective/
|
|-- README.txt
|-- pyproject.toml / requirements.txt <- It has the modules I used and version.
|
|-- data/                    <- Here I stored the sample data and the .parquet that I saved with the clean data, but it would be partitioned as I stated earlier
|   |-- cleaned_data_for_fcst.parquet   <- Cleaned data that I exported
|   |-- sample_data__technical_assessment_1.xlsx   <- Sample data
|   |-- .gitkeep
|
|-- Code/               <- exploratory & presentation notebooks (for this exersice I'll upload .py files since I coded on pure python)
|   |-- Data Exploration.py
|   |-- Functions.py
|
|-- src/                     <- pure Python code importable as a package
|   |-- __init__.py
|   |-- features/            <- deterministic feature logic
|   |   |-- __init__.py
|   |   |-- make_dataset.py  <- This doesn't exist
|   |
|   |-- models/              <- training scripts & model defs
|   |   |-- churn_model_oop.py   # OOP artefact creator
|   |   |-- train.py             # thin wrapper to call EarlyChurnModel via CLI
|   |	|-- Baseline training.py
|   |	|-- Early Churn Classifier.py
|   |
|   |-- monitoring/         <- production health checks
|       |-- monitoring_pipeline.py
|
|-- configs/                 <- partner-specific YAMLs
|   |-- AZ01.yaml
|   |-- CO02.yaml
|   |-- …
|
|-- docker/                  <- Dockerfile + entrypoints
|
|-- artifacts/               <- Model outputs only
|   |-- AZ01/
|   |   |-- pipeline.joblib
|   |   |-- meta.json
|   |   |-- metrics.json
|   |-- km_baseline_survival.csv
|
|-- .github/                 <- workflow templates


