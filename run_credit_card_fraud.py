from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import expr, avg


def main():
    spark = SparkSession.builder.appName("CreditCardFraud").getOrCreate()

    df = (
        spark.read
        .option("inferSchema", True)
        .option("header", True)
        .csv("data/creditcard-fraud.csv")
    )

    print("Schema:")
    df.printSchema()
    print("Sample data:")
    df.show(10, truncate=False)

    feature_columns = [col for col in df.columns if col.startswith("V")] + ["Amount"]

    if not feature_columns:
        raise ValueError("No numeric feature columns found in the dataset.")

    print("Feature columns:", feature_columns)

    vectorizer = VectorAssembler(inputCols=feature_columns, outputCol="features")
    rf = RandomForestClassifier(labelCol="Class", maxDepth=5)
    pipeline = Pipeline(stages=[vectorizer, rf])

    df_train, df_test = df.randomSplit(weights=[0.7, 0.3], seed=1)
    model = pipeline.fit(df_train)
    df_test_pred = model.transform(df_test)

    label_counts = df.groupBy("Class").count().orderBy("Class").collect()
    print("Label distribution:")
    for row in label_counts:
        print(f"  {row['Class']}: {row['count']}")

    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="Class", metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(df_test_pred)

    f1_evaluator = MulticlassClassificationEvaluator(labelCol="Class", metricName="f1")
    f1_score = f1_evaluator.evaluate(df_test_pred)

    test_accuracy = (
        df_test_pred
        .select("Class", "prediction")
        .withColumn("isEqual", expr("Class == prediction"))
        .select(avg(expr("cast(isEqual as float)")))
        .first()[0]
    )

    binary_evaluator = BinaryClassificationEvaluator(labelCol="Class")
    auc = binary_evaluator.evaluate(df_test_pred)

    print(f"AUC: {auc}")
    print(f"Accuracy (MulticlassEvaluator): {accuracy}")
    print(f"F1 score: {f1_score}")
    print(f"Accuracy (direct compare): {test_accuracy}")

    spark.stop()


if __name__ == "__main__":
    main()
