import os
import sys

# Ép PySpark sử dụng đúng môi trường Python (Miniconda) hiện tại
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, sum

# 1. Khởi tạo SparkSession
spark = SparkSession.builder.appName("PySpark_Basic_Exercise").getOrCreate()

# 2. Tạo DataFrame Bán hàng (Sales)
sales_data = [
    (1, "Nguyen Van A", "iPhone 14", 2, 800),
    (2, "Tran Thi B", "Samsung S23", 1, 900),
    (3, "Nguyen Van A", "AirPods Pro", 1, 250),
    (4, "Le Van C", "iPhone 14", 1, 800),
    (5, "Tran Thi B", "iPad Air", 2, 600)
]
sales_columns = ["order_id", "customer_name", "product_name", "quantity", "price_per_unit"]
df_sales = spark.createDataFrame(data=sales_data, schema=sales_columns)

# 3. Tạo DataFrame Danh mục Sản phẩm (Products)
category_data = [
    ("iPhone 14", "Điện thoại"),
    ("Samsung S23", "Điện thoại"),
    ("AirPods Pro", "Phụ kiện"),
    ("iPad Air", "Máy tính bảng")
]
category_columns = ["product_name", "category"]
df_category = spark.createDataFrame(data=category_data, schema=category_columns)

print("====== Câu 1 ======")
df_sales.printSchema()
df_sales.show()

print("====== Câu 2 ======")
df_sales_total = df_sales.withColumn("total_amount", col("quantity") * col("price_per_unit"))
df_sales_total.show()

print("====== Câu 3 ======")
df_high_value = df_sales_total.filter(df_sales_total.total_amount >= 1000)
df_high_value.show()

print("====== Câu 4 ======")
df_customer_spend = df_sales_total.groupBy(df_sales_total.customer_name) \
    .agg(sum(df_sales_total.total_amount).alias("total_spent")) \
    .orderBy(col("total_spent").desc())
df_customer_spend.show()

print("====== Câu 5 ======")
df_joined = df_sales_total.join(df_category, on="product_name", how="left")
df_joined.show()

print("====== Câu 6 ======")
df_category_revenue = df_joined.groupBy("category") \
    .agg(sum("total_amount").alias("total_revenue")) \
    .orderBy(col("total_revenue").desc())
df_category_revenue.show(1)