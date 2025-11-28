from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, regexp_extract, avg, round

# ------------------------------
# 1. Start Spark
# ------------------------------
spark = SparkSession.builder \
    .appName("Healthcare_ETL_Analytics") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# ------------------------------
# 2. Load datasets
# ------------------------------
patients_df = spark.read.csv("data/healthcare_dataset.csv", header=True, inferSchema=True)
readmissions_df = spark.read.csv("data/hospital_readmissions.csv", header=True, inferSchema=True)

# ------------------------------
# 3. Normalize column names
# ------------------------------
patients_df = patients_df.toDF(*[c.lower().replace(" ", "_") for c in patients_df.columns])
readmissions_df = readmissions_df.toDF(*[c.lower().replace(" ", "_") for c in readmissions_df.columns])

# ------------------------------
# 4. Create patient_id for patients
# ------------------------------
patients_df = patients_df.withColumn("patient_id", monotonically_increasing_id())

# ------------------------------
# 5. Transform readmissions age to numeric
# ------------------------------
readmissions_df = readmissions_df.withColumn(
    "age_lower", regexp_extract(col("age"), r"\[(\d+)-\d+\)", 1).cast("integer")
)

# ------------------------------
# 6. Aliases for join
# ------------------------------
p = patients_df.alias("p")
r = readmissions_df.alias("r")

# ------------------------------
# 7. Join datasets (patients.age between readmissions.age_lower and age_lower+9)
# ------------------------------
joined_df = p.join(
    r,
    (col("p.age").between(col("r.age_lower"), col("r.age_lower") + 9)),
    how="inner"
)

# ------------------------------
# 8. Select columns with aliases to avoid ambiguity
# ------------------------------
final_df = joined_df.select(
    col("p.patient_id"),
    col("p.name"),
    col("p.age").alias("age"),
    col("p.gender"),
    col("p.blood_type"),
    col("p.medical_condition"),
    col("p.date_of_admission"),
    col("p.doctor"),
    col("p.hospital"),
    col("p.insurance_provider"),
    col("p.billing_amount"),
    col("p.room_number"),
    col("p.admission_type"),
    col("p.discharge_date"),
    col("p.medication"),
    col("p.test_results"),
    col("r.time_in_hospital"),
    col("r.n_lab_procedures"),
    col("r.n_procedures"),
    col("r.n_medications"),
    col("r.n_outpatient"),
    col("r.n_inpatient"),
    col("r.n_emergency"),
    col("r.medical_specialty"),
    col("r.diag_1"),
    col("r.diag_2"),
    col("r.diag_3"),
    col("r.glucose_test"),
    col("r.a1ctest"),
    col("r.change"),
    col("r.diabetes_med"),
    col("r.readmitted")
)

# ------------------------------
# 9. Show sample and row count
# ------------------------------
print("\n=== Sample Joined Dataset ===")
final_df.show(5, truncate=False)
print("\nTotal rows in joined dataset:", final_df.count())
print("Total unique patients:", final_df.select("patient_id").distinct().count())

# ------------------------------
# 10. Basic Analytics / Reporting
# ------------------------------

# Gender Distribution
print("\n=== Gender Distribution ===")
final_df.groupBy("gender").count().show()

# Age Statistics
print("\n=== Average Age ===")
final_df.select(round(avg("age"),2).alias("avg_age")).show()

# Top 5 Hospitals
print("\n=== Top 5 Hospitals by Patient Count ===")
final_df.groupBy("hospital").count().orderBy(col("count").desc()).show(5, truncate=False)

# Top 5 Doctors
print("\n=== Top 5 Doctors by Patient Count ===")
final_df.groupBy("doctor").count().orderBy(col("count").desc()).show(5, truncate=False)

# Readmission Counts
print("\n=== Readmission Counts ===")
final_df.groupBy("readmitted").count().show()

# Average Time in Hospital by Readmission
print("\n=== Average Time in Hospital by Readmission ===")
final_df.groupBy("readmitted").agg(round(avg("time_in_hospital"),2).alias("avg_days")).show()

# Average Billing by Admission Type
print("\n=== Average Billing by Admission Type ===")
final_df.groupBy("admission_type").agg(round(avg("billing_amount"),2).alias("avg_billing")).show()

# ------------------------------
# 11. Save final dataset to Parquet (optional)
# ------------------------------
final_df.write.mode("overwrite").parquet("output/final_healthcare_etl.parquet")

# ------------------------------
# 12. Stop Spark
# ------------------------------
spark.stop()
