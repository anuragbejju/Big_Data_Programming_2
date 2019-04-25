#  Professional Masters in Big Data Program - Simon Fraser University

#  Assignment 2 (Question 1 - entity_resolution.py)

#  Submission Date: 20th January 2019
#  Name: Anurag Bejju
#  Student ID: 301369375

# Running Instructions:

# Default Run: ${SPARK_HOME}/bin/spark-submit entity_resolution.py    ==> [Threshold = 0.5, File = amazon-google-sample]
# Set Threshold on Sample Set: ${SPARK_HOME}/bin/spark-submit entity_resolution.py amazon-google-sample 0.5   ==> [Threshold = 0.5, File = amazon-google-sample]
# Run on larger Set: ${SPARK_HOME}/bin/spark-submit entity_resolution.py amazon-google 0.5   ==> [Threshold = 0.5, File = amazon-google]

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,array,col,concat_ws,lower
from pyspark.sql.types import StringType,ArrayType, FloatType
from pyspark.ml.feature import StopWordsRemover
import re,sys


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    length_set1 = len(set1)
    length_set2 = len(set2)
    interset_between_two_sets = set1.intersection(set2)
    length_interset_between_two_sets = len(interset_between_two_sets)
    jaccard_value = length_interset_between_two_sets / (length_set1 + length_set2 - length_interset_between_two_sets)
    return jaccard_value

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = list(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = spark.read.parquet(dataFile1).cache()
        self.df2 = spark.read.parquet(dataFile2).cache()

    def preprocessDF(self, df, cols):
        #add null string to stop word list
        stop_words = self.stopWordsBC
        stop_words.append('')

        # concatenate the $cols and lower-case it
        df_concatenate = df.withColumn("concatenate_value", lower(concat_ws(' ',*cols)))

        # apply the tokenizer to the concatenated string
        tokenize_string = udf((lambda str: re.split(r'\W+', str)), ArrayType(StringType()))
        df_tokenized = df_concatenate.withColumn('tokens', tokenize_string(df_concatenate.concatenate_value))

        # Remove stop words using StopWordsRemover
        stopWords_remover = StopWordsRemover(inputCol="tokens", outputCol="joinKey", stopWords=stop_words)
        df = stopWords_remover.transform(df_tokenized).drop("concatenate_value").drop("tokens")
        return df

    def filtering(self, df1, df2):

        # select id and joinKey from both the dataframes
        df1 = df1.select(col("id").alias("id1"), col("joinKey").alias("joinKey1")).cache()
        df2 = df2.select(col("id").alias("id2"), col("joinKey").alias("joinKey2")).cache()

        # Convert to rdd and the id mapped to each token in the list
        rdd1_df1 = df1.rdd.flatMapValues(tuple).toDF(["id1","join_token1"]).distinct()
        rdd1_df1.createOrReplaceTempView("flat_table1")
        rdd2_df2 = df2.rdd.flatMapValues(tuple).toDF([ "id2","join_token2"]).distinct()
        rdd2_df2.createOrReplaceTempView("flat_table2")

        # get a mapping between the two ids based on join_tokens
        mapped_df = spark.sql("select distinct t1.id1, t2.id2 from flat_table1 as t1, flat_table2 as t2 where t1.join_token1 = t2.join_token2").cache()

        df1.createOrReplaceTempView("table1")
        df2.createOrReplaceTempView("table2")
        mapped_df.createOrReplaceTempView("table3")

        # Finally join all the three tables using the indexes in mapped_df
        output_df = spark.sql("select t3.id1, t3.id2, t1.joinKey1, t2.joinKey2 from table1 as t1, table2 as t2, table3 as t3 where t3.id2 = t2.id2 and t3.id1 = t1.id1")
        return output_df

    def verification(self, candDF, threshold):

        #Caluclate the jaccard similarity between $joinKey1 and $joinKey2
        jaccard_cal = udf(jaccard_similarity, FloatType())
        df = candDF.withColumn('jaccard', jaccard_cal(candDF.joinKey1,candDF.joinKey2))

        # Remove the rows whose jaccard similarity is smaller than $threshold
        return df.filter(df.jaccard >= threshold)

    def evaluate(self, result, groundTruth):

        # Calculate precision, recall and fmeasure

        R = len(list(set(result).intersection(groundTruth)))
        T = len(result)
        A = len(groundTruth)

        precision = R/T
        recall = R/A
        fmeasure = (2*precision*recall)/(precision+recall)
        return (precision, recall, fmeasure)

    def jaccardJoin(self, cols1, cols2, threshold):

        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()))

        candDF = self.filtering(newDF1, newDF2)
        print ("After Filtering: %d pairs left" %(candDF.count()))

        resultDF = self.verification(candDF, threshold)
        print ("After Verification: %d similar pairs" %(resultDF.count()))
        return resultDF


    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    conf = SparkConf().setAppName('ER Application')
    sc = SparkContext(conf=conf)
    spark = SQLContext(sc)

    if (len(sys.argv) >= 2):
        file = sys.argv[1]
        if file == 'amazon-google-sample':
            amazon_file = file + "/Amazon_sample"
            google_file = file + "/Google_sample"
            stopWords_file = file + "/stopwords.txt"
            perfectMapping_file = file + "/Amazon_Google_perfectMapping_sample"

        elif file == 'amazon-google':
            amazon_file = file + "/Amazon"
            google_file = file + "/Google"
            stopWords_file = file + "/stopwords.txt"
            perfectMapping_file = file + "/Amazon_Google_perfectMapping"

        else:
            raise Exception('File not found!')
    elif (len(sys.argv) == 1):
        file = 'amazon-google-sample'
        amazon_file = file + "/Amazon_sample"
        google_file = file + "/Google_sample"
        stopWords_file = file + "/stopwords.txt"
        perfectMapping_file = file + "/Amazon_Google_perfectMapping_sample"

    if (len(sys.argv) == 3):
        threshold = sys.argv[2]
    else:
        threshold = 0.5

    sc.setLogLevel('WARN')
    er = EntityResolution(amazon_file, google_file, stopWords_file)

    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, threshold)
    groundTruth = spark.read.parquet(perfectMapping_file) \
                         .rdd.map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect()

    print ("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))
