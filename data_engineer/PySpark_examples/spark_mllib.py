# Required imports (Sử dụng pyspark.ml thay cho mllib)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ML_DataFrame_Example").getOrCreate()

# Load and split data (Bỏ .rdd đi để giữ nguyên định dạng DataFrame)
data = spark.read.csv("data.csv", header=True, inferSchema=True)
train_data, test_data = data.randomSplit([0.7, 0.3])

# Classification - Logistic Regression
def logistic_regression(train_data, test_data):
    # Cần chỉ định rõ cột features và label
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_data)
    predictions = model.transform(test_data)
    return predictions

# Regression - Linear Regression
def linear_regression(train_data, test_data):
    lr = LinearRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_data)
    predictions = model.transform(test_data)
    return predictions

# Clustering - KMeans
def kmeans_clustering(data):
    kmeans = KMeans(featuresCol="features", k=3)
    model = kmeans.fit(data)
    predictions = model.transform(data)
    return predictions

# Collaborative Filtering - ALS
def collaborative_filtering(train_data, test_data):
    # Cần chỉ định các cột tương ứng trong file dữ liệu của bạn
    als = ALS(maxIter=10, rank=10, userCol="userId", itemCol="productId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(train_data)
    predictions = model.transform(test_data)
    return predictions

# Data Preprocessing - StandardScaler
def scale_features(data):
    # Lưu ý: Cột "features" truyền vào phải được vectorize bằng VectorAssembler trước đó
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    scaled_data = scaler.fit(data).transform(data)
    return scaled_data

# Data Preprocessing - StringIndexer (for categorical variables)
def index_categorical_features(data):
    indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
    indexed_data = indexer.fit(data).transform(data)
    return indexed_data

# Model Evaluation - MulticlassClassificationEvaluator
def evaluate_model(predictions):
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    return accuracy