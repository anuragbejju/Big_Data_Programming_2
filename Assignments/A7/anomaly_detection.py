#  Professional Masters in Big Data Program - Simon Fraser University

#  Assignment 7 (Question 1 - anomaly_detection.py)

#  Submission Date: 2nd March 2019
#  Name: Anurag Bejju
#  Student ID: 301369375

# Running instruction : spark-submit anomaly_detection.py filename
# Defualt: spark-submit anomaly_detection.py data/logs-features-sample

import os
import sys
from pyspark import SparkConf, SparkContext, SQLContext

conf = SparkConf().setAppName('Assignment 7')
sc = SparkContext(conf=conf)
spark = SQLContext(sc)
sc.setLogLevel("WARN")

from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline


class AnomalyDetection():
    def readToyData(self):
        data = [(0, ["http", "udt", 0.4]), (1, ["http", "udf", 0.5]),
                (2, ["http", "tcp", 0.5]), (3, ["ftp", "icmp", 0.1]),
                (4, ["http", "tcp", 0.4])]
        schema = ["id", "rawFeatures"]
        self.rawDF = spark.createDataFrame(data, schema)

    def readData(self, filename):
        # Reading parquet file
        self.rawDF = spark.read.parquet(filename)

    def cat2Num(self, df, indices):

        # Function to one hot encode catergorical columns
        def one_hot_encoder(rawFeatures, indices, category_set):
            features_array = []
            index = 0

            # Iterate through each rawFeatures and check if its index is a categorical feature or not
            # Then create an array of zero length and only change it to 1 where the category index match
            for feature in rawFeatures:
                if index in indices:
                    unique_feature_set = category_set[index]
                    feature_index = unique_feature_set.index(feature)
                    one_hot_encoding = [0] * (len(unique_feature_set))
                    one_hot_encoding[feature_index] = 1
                    features_array = features_array + one_hot_encoding
                else:
                    features_array.append(feature)
                index = index + 1
            return features_array

        # Split all category values from rawFeatures list
        df_cat_split = df.select("id", "rawFeatures",
                                 *[(df["rawFeatures"][i]).alias(str(i))
                                   for i in indices])
        # Collect the unique list for each categorical values
        category_set_expr = [collect_set(str(colName)) for colName in indices]
        df_collect_set = df_cat_split.agg(*category_set_expr)

        # Create a collection of all categorical vaues
        category_set = df_collect_set.collect()[0]

        # Pass the rawFeatures, indices and category_set to UDF function to one-hot encode it
        udf_one_hot_encoder = udf(
            lambda rawFeatures: one_hot_encoder(rawFeatures, indices, category_set),
            ArrayType(StringType()))
        encoded_df = df_cat_split.withColumn("features",
                                             udf_one_hot_encoder(
                                                 df_cat_split["rawFeatures"]))
        df = encoded_df.select("id", "rawFeatures", "features")
        return df

    def addScore(self, df):
        def calc_score(count, min_max_collection):
            min_count = float(min_max_collection[0])
            max_count = float(min_max_collection[1])
            score = (max_count - count) / (max_count - min_count)
            return score

        # get the count for each prediction value
        count_df = df.groupBy("prediction").count().cache()

        # get the min and max of the count as a collection and pass to udf
        min_max_collection = count_df.agg(
            min('count').alias('min'),
            max('count').alias('max')).collect()[0]

        # calculate the score and append it to count_df
        udf_calc_score = udf(
            lambda count: calc_score(float(count), min_max_collection),
            FloatType())
        scored_df = count_df.withColumn("score",
                                        udf_calc_score("count")).cache()

        # Join the scored to to our original df
        df = df.join(scored_df, "prediction").select(
            "id", "rawFeatures", "features", "prediction", "score")
        return df

    def detect(self, k, t):
        print('\x1b[1;31m', 'Start One-hot encoding catergorical features.',
              '\x1b[0m')
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show(10)
        print('\x1b[1;31m',
              'Completed One-hot encoding catergorical features.', '\x1b[0m')

        #Clustering points using KMeans
        print('\x1b[1;31m', 'Clustering points using KMeans', '\x1b[0m')
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(
            features,
            k,
            maxIterations=40,
            runs=10,
            initializationMode="random",
            seed=20)
        #Adding the prediction column to df1
        modelBC = sc.broadcast(model)

        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show(10)
        print('\x1b[1;31m', 'Completed clustering points using KMeans',
              '\x1b[0m')

        #Adding the score column to df2; The higher the score, the more likely it is an anomaly
        print('\x1b[1;31m', 'Adding anomaly score', '\x1b[0m')
        df3 = self.addScore(df2).cache()
        df3.show(10)
        print('\x1b[1;31m', 'Completed!', '\x1b[0m')
        return df3.where(df3.score > t)


if __name__ == "__main__":
    ad = AnomalyDetection()
    if (len(sys.argv) >= 2):
        inputs = sys.argv[1]
    else:
        inputs = 'data/logs-features-sample'
    ad.readData(inputs)
    #ad.readToyData()
    #anomalies = ad.detect(2, 0.9)
    anomalies = ad.detect(8, 0.97)
    print('\x1b[1;31m', 'Total number of anomalies in the given dataset are: ',
          '\x1b[0m', anomalies.count())
    anomalies.show(10)
