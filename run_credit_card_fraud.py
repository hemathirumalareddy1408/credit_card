import argparse
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import expr, avg


def build_spark_session(app_name="CreditCardFraud"):
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_data(spark, data_path):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
        .cache()
    )


def get_feature_columns(df):
    return [col for col in df.columns if col.startswith("V")] + ["Amount"]


def print_confusion_matrix(df_test_pred, label_col="Class"):
    print("Confusion matrix:")
    for row in (
        df_test_pred
        .groupBy(label_col, "prediction")
        .count()
        .orderBy(label_col, "prediction")
        .collect()
    ):
        print(f"  Actual={row[label_col]}, Predicted={row['prediction']}, Count={row['count']}")


def evaluate_model(df_test_pred, label_col="Class"):
    evaluators = {
        "Accuracy": MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy"),
        "F1 score": MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1"),
        "AUC": BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC"),
    }
    results = {name: evaluator.evaluate(df_test_pred) for name, evaluator in evaluators.items()}
    direct_accuracy = (
        df_test_pred
        .select(label_col, "prediction")
        .withColumn("isEqual", expr(f"{label_col} == prediction"))
        .select(avg(expr("cast(isEqual as float)")))
        .first()[0]
    )
    results["Accuracy (direct compare)"] = direct_accuracy
    return results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Spark ML model for credit card fraud detection."
    )
    parser.add_argument(
        "--data-path",
        default="data/creditcard-fraud.csv",
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Fraction of data used for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for train/test split.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    spark = build_spark_session()
    df = load_data(spark, args.data_path)

    print("Schema:")
    df.printSchema()
    print("Sample data:")
    df.show(10, truncate=False)

    feature_columns = get_feature_columns(df)
    if not feature_columns:
        raise ValueError("No numeric feature columns found in the dataset.")
    print("Feature columns:", feature_columns)

    vectorizer = VectorAssembler(inputCols=feature_columns, outputCol="features")
    rf = RandomForestClassifier(labelCol="Class", featuresCol="features", maxDepth=5)
    pipeline = Pipeline(stages=[vectorizer, rf])

    df_train, df_test = df.randomSplit([1.0 - args.test_fraction, args.test_fraction], seed=args.seed)
    model = pipeline.fit(df_train)
    df_test_pred = model.transform(df_test)

    label_counts = df.groupBy("Class").count().orderBy("Class").collect()
    print("Label distribution:")
    for row in label_counts:
        print(f"  {row['Class']}: {row['count']}")

    print_confusion_matrix(df_test_pred)
    results = evaluate_model(df_test_pred)
    for name, value in results.items():
        print(f"{name}: {value}")

    spark.stop()


if __name__ == "__main__":
    main()
