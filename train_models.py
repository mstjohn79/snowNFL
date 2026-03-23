import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import snowflake.connector
from snowflake.snowpark import Session
from snowflake.ml.registry import Registry

CONN_NAME = os.getenv("SNOWFLAKE_CONNECTION_NAME") or "martydemo"
DB = "NFL_ANALYTICS"
SCHEMA = "OL_SCORING"

def get_connector():
    conn = snowflake.connector.connect(connection_name=CONN_NAME)
    conn.cursor().execute(f"USE DATABASE {DB}")
    conn.cursor().execute(f"USE SCHEMA {SCHEMA}")
    conn.cursor().execute("USE WAREHOUSE COMPUTE_WH")
    return conn

def get_snowpark_session():
    return Session.builder.configs({"connection_name": CONN_NAME}).create()

FEATURE_COLS = [
    "SACK_RATE", "QB_HIT_RATE", "PRESSURE_RATE", "AVG_TIME_TO_THROW",
    "PASS_EPA_PER_PLAY", "PASS_SUCCESS_RATE", "SCRAMBLE_RATE",
    "STUFF_RATE", "AVG_RUSH_YARDS", "RUSH_EPA_PER_PLAY",
    "RUSH_SUCCESS_RATE", "EXPLOSIVE_RUN_RATE", "SHORT_GAIN_RATE",
    "AVG_DEFENDERS_IN_BOX", "OL_PENALTIES", "HOLDING_PENALTIES",
    "FALSE_START_PENALTIES", "DROPBACKS", "RUSH_ATTEMPTS",
    "EARLY_DOWN_EPA", "LATE_DOWN_EPA", "EARLY_VS_LATE_EPA_DIFF",
    "THIRD_SHORT_SUCCESS", "THIRD_LONG_SACK_RATE", "THIRD_LONG_PRESSURE_RATE",
    "REDZONE_EPA", "REDZONE_STUFF_RATE", "REDZONE_SACK_RATE",
    "OWN_TERRITORY_EPA", "SITUATIONAL_EPA_VARIANCE",
    "ROLLING_3_OL_SCORE", "ROLLING_5_OL_SCORE",
    "ROLLING_3_PASS_BLOCK", "ROLLING_3_RUN_BLOCK",
    "ROLLING_5_SACK_RATE", "ROLLING_5_PRESSURE_RATE",
    "PREV_GAME_OL_SCORE",
    "ROLLING_3_REDZONE_EPA", "ROLLING_3_CLUTCH_DIFF", "ROLLING_3_SIT_VARIANCE",
    "OL_CLUTCH_INDEX", "OL_CLEAN_POCKET_RATE", "OL_SUCCESS_RATE",
    "OL_HIGH_LEVERAGE_EPA", "OL_PROTECTION_CLUTCH", "OL_RUN_BLOCKING_CLUTCH",
    "ROLLING_3_CLUTCH_INDEX", "ROLLING_3_CLEAN_POCKET", "ROLLING_3_PROTECTION_CLUTCH",
]

def load_data():
    print("Loading ML_TRAINING_DATA from Snowflake...")
    conn = get_connector()
    df = pd.read_sql("SELECT * FROM ML_TRAINING_DATA", conn)
    conn.close()
    df.columns = [c.upper() for c in df.columns]
    df = df.dropna(subset=FEATURE_COLS + ["TARGET_OL_SCORE", "TARGET_PASS_BLOCK", "TARGET_RUN_BLOCK"])
    df = df.sort_values(["SEASON", "WEEK"]).reset_index(drop=True)
    print(f"  Loaded {len(df)} rows after dropping nulls")
    return df

def train_xgb_model(X_train, y_train, X_test, y_test, name):
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n  {name} Results:")
    print(f"    MAE:  {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    R2:   {r2:.4f}")
    return model, {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}

def main():
    df = load_data()

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]

    print("\n=== Training Composite OL Score Model ===")
    ol_model, ol_metrics = train_xgb_model(
        X_train, train_df["TARGET_OL_SCORE"],
        X_test, test_df["TARGET_OL_SCORE"],
        "Composite OL Score"
    )

    print("\n=== Training Pass Block Score Model ===")
    pass_model, pass_metrics = train_xgb_model(
        X_train, train_df["TARGET_PASS_BLOCK"],
        X_test, test_df["TARGET_PASS_BLOCK"],
        "Pass Block Score"
    )

    print("\n=== Training Run Block Score Model ===")
    run_model, run_metrics = train_xgb_model(
        X_train, train_df["TARGET_RUN_BLOCK"],
        X_test, test_df["TARGET_RUN_BLOCK"],
        "Run Block Score"
    )

    print("\n=== Registering Models in Snowflake Model Registry ===")
    session = get_snowpark_session()
    session.sql(f"USE DATABASE {DB}").collect()
    session.sql(f"USE SCHEMA {SCHEMA}").collect()
    session.sql("USE WAREHOUSE COMPUTE_WH").collect()

    reg = Registry(session=session, database_name=DB, schema_name=SCHEMA)

    sample_input = session.create_dataframe(X_test.head(10))

    print("  Registering composite OL model...")
    mv_ol = reg.log_model(
        ol_model,
        model_name="NFL_OL_COMPOSITE_PREDICTOR",
        version_name="v4",
        conda_dependencies=["xgboost", "scikit-learn"],
        sample_input_data=sample_input,
        comment="XGBoost v4 with corrected percentile scoring (lower-is-better metrics fixed)",
        metrics=ol_metrics,
    )
    print(f"    Registered: NFL_OL_COMPOSITE_PREDICTOR v4")

    print("  Registering pass block model...")
    mv_pass = reg.log_model(
        pass_model,
        model_name="NFL_PASS_BLOCK_PREDICTOR",
        version_name="v4",
        conda_dependencies=["xgboost", "scikit-learn"],
        sample_input_data=sample_input,
        comment="XGBoost v4 pass block with corrected percentile scoring",
        metrics=pass_metrics,
    )
    print(f"    Registered: NFL_PASS_BLOCK_PREDICTOR v4")

    print("  Registering run block model...")
    mv_run = reg.log_model(
        run_model,
        model_name="NFL_RUN_BLOCK_PREDICTOR",
        version_name="v4",
        conda_dependencies=["xgboost", "scikit-learn"],
        sample_input_data=sample_input,
        comment="XGBoost v4 run block with corrected percentile scoring",
        metrics=run_metrics,
    )
    print(f"    Registered: NFL_RUN_BLOCK_PREDICTOR v4")

    print("\n=== Running Inference on Test Data ===")
    test_sp = session.create_dataframe(X_test)
    ol_preds = mv_ol.run(test_sp, function_name="predict")
    pass_preds = mv_pass.run(test_sp, function_name="predict")
    run_preds = mv_run.run(test_sp, function_name="predict")

    print(f"  OL predictions shape: {ol_preds.count()} rows")
    print(f"  Pass block predictions shape: {pass_preds.count()} rows")
    print(f"  Run block predictions shape: {run_preds.count()} rows")

    print("\n=== Feature Importance (Top 15 for Composite OL) ===")
    importances = ol_model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:15]:
        print(f"    {feat}: {imp:.4f}")

    print("\n=== Creating Prediction Tables ===")
    feature_col_sql = ", ".join(FEATURE_COLS)
    session.sql(f"""
        CREATE OR REPLACE TABLE ML_PREDICTIONS AS
        WITH latest_features AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY TEAM ORDER BY SEASON DESC, WEEK DESC) AS rn
            FROM ML_TRAINING_DATA
        )
        SELECT TEAM, SEASON, WEEK, {feature_col_sql}
        FROM latest_features
        WHERE rn = 1
    """).collect()

    latest_features = session.table("ML_PREDICTIONS")
    ol_latest = mv_ol.run(latest_features.select(FEATURE_COLS), function_name="predict")
    pass_latest = mv_pass.run(latest_features.select(FEATURE_COLS), function_name="predict")
    run_latest = mv_run.run(latest_features.select(FEATURE_COLS), function_name="predict")

    print("  Inference complete on latest features")

    meta_cols = latest_features.select("TEAM", "SEASON", "WEEK").to_pandas()
    ol_pdf = ol_latest.to_pandas()
    pass_pdf = pass_latest.to_pandas()
    run_pdf = run_latest.to_pandas()

    pred_col = [c for c in ol_pdf.columns if "output" in c.lower() or "predict" in c.lower()]
    ol_col_name = pred_col[0] if pred_col else ol_pdf.columns[-1]
    pred_col = [c for c in pass_pdf.columns if "output" in c.lower() or "predict" in c.lower()]
    pass_col_name = pred_col[0] if pred_col else pass_pdf.columns[-1]
    pred_col = [c for c in run_pdf.columns if "output" in c.lower() or "predict" in c.lower()]
    run_col_name = pred_col[0] if pred_col else run_pdf.columns[-1]

    result = meta_cols.copy()
    result["PREDICTED_OL_SCORE"] = ol_pdf[ol_col_name].values
    result["PREDICTED_PASS_BLOCK"] = pass_pdf[pass_col_name].values
    result["PREDICTED_RUN_BLOCK"] = run_pdf[run_col_name].values

    result_sp = session.create_dataframe(result)
    result_sp.write.mode("overwrite").save_as_table("ML_NEXT_GAME_PREDICTIONS")

    print("\n=== Next-Game Predictions (Top 10 OL Score) ===")
    top10 = result.sort_values("PREDICTED_OL_SCORE", ascending=False).head(10)
    for _, row in top10.iterrows():
        print(f"    {row['TEAM']}: OL={row['PREDICTED_OL_SCORE']:.1f}  Pass={row['PREDICTED_PASS_BLOCK']:.1f}  Run={row['PREDICTED_RUN_BLOCK']:.1f}")

    session.close()
    print("\nAll models trained, registered, and predictions generated!")

if __name__ == "__main__":
    main()
