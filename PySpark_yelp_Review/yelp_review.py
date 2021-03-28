import pandas as pd 
import numpy as np 
import re
import findspark
import os

os.environ["JAVA_HOME"] = "D:/Java/jdk8"
os.environ["SPARK_HOME"] = "D:/Pyspark/spark"

findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession

conf = SparkConf().setAppName("MyApp").set('spark.driver.memory', '8g').set('spark.executor.memory', '8g')

sc = SparkContext(conf = conf)
spark = SQLContext(sc)
dfcsv = spark.read.csv(r'F:/Baruch College/My courses/2020 Spring/STA 9760/Final Project/yelp_review.csv', header = True)
doc = dfcsv.select(dfcsv.text).collect()
print(doc.first())

# doc = sc.textFile("F:/Baruch College/My courses/2020 Spring/STA 9760/midterm/yelp_review_revised.txt")

# flattened = doc.filter(lambda line: len(line) > 0).flatMap(lambda line: re.split("W+", line))

# kvPairs = flattened.filter(lambda word : len(word) > 3).map(lambda word: (word.lower(), 1))

# countsByWord = kvPairs.reduceByKey(lambda v1, v2 : v1+v2).sortByKey(ascending = False)

# topWords = countsByWord.map(lambda x: (x[1], x[0])).sortByKey(ascending = False)

# topWords.take(10)

# # the output is showed below:

# # [(6318, 'cons:'), 
# #  (4676, 'pros:'), 
# #  (3126, 'e will be back!"'),
# #  (2563, 'ild '), 
# #  (2347, 'e will be back."'), 
# #  (2320, 'food:'), 
# #  (2196, 'e will definitely be back!"'), 
# #  (1828, '"the '), 
# #  (1745, 'the '), 
# #  (1726, 'enjoy!"')]
