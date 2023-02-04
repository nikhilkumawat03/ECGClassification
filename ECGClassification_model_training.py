# Databricks notebook source
# import findspark
# findspark.init()

# from pyspark.context import SparkContext, SparkConf
# from pyspark.sql import SparkSession

# COMMAND ----------

# sql_context = SparkSession.builder.config("spark.executor.memory", '2g') \
#                                     .config('spark.executor.cores', '2') \
#                                     .config("spark.driver.memory", '4g') \
#                                     .master('local[*]').getOrCreate()


# COMMAND ----------

# sql_context.sparkContext.setLogLevel("OFF")

# COMMAND ----------

# sql_context

# COMMAND ----------

import numpy as np 
import pandas as pd 
import os

import matplotlib.pyplot as plt
import csv
import itertools
import collections

import pywt
from scipy import stats

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# COMMAND ----------

plt.rcParams["figure.figsize"] = (30,6)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True 

# COMMAND ----------

def denoise(data): 
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04 # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
        
    datarec = pywt.waverec(coeffs, 'sym4')
    
    return datarec

# COMMAND ----------

#path='/dbfs/FileStore/tables/mitbihDatabase/'
s3_bucket = 
path='s3://datasciencehubbucket/mitbih_database/'
window_size = 140
maximum_counting = 10000

classes = [['N', 'L', 'R', 'e', 'j'], ['A', 'a', 'J', 'S'], ['V', 'E'], ['F'], ['Q']]
n_classes = len(classes)
count_classes = [0]*n_classes

X = list()
y = list()

# COMMAND ----------

# Read files
import os
filenames = os.listdir(path)
# Split and save .csv , .txt 
records = list()
annotations = list()
filenames.sort()
# segrefating filenames and annotations
for f in filenames:
    #filename, file_extension = f.split('.')
    filename, file_extension = os.path.splitext(f)
 # *.csv
    if(file_extension == '.csv'):
        records.append(path + filename + file_extension)

    # *.txt
    else:
        annotations.append(path + filename + file_extension)
print(records)

# COMMAND ----------

# Records
for r in range(0,len(records)):
    signals = []

    with open(records[r], 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|') # read CSV file\
        row_index = -1
        for row in spamreader:
            if(row_index >= 0):
                signals.insert(row_index, int(row[1]))
            row_index += 1
            
    # Plot an example to the signals
    if r is 1:
        # Plot each patient's signal
        plt.title(records[1] + " Wave")
        plt.plot(signals[0:700])
        plt.show()
        
    signals = denoise(signals)
    # Plot an example to the signals
    if r is 1:
        # Plot each patient's signal
        plt.title(records[1] + " wave after denoised")
        plt.plot(signals[0:700])
        plt.show()
        
    signals = stats.zscore(signals)
    # Plot an example to the signals
    if r is 1:
        # Plot each patient's signal
        plt.title(records[1] + " wave after z-score normalization ")
        plt.plot(signals[0:700])
        plt.show()
    
    # Read anotations: R position and Arrhythmia class
    example_beat_printed = False
    with open(annotations[r], 'r') as fileID:
        data = fileID.readlines() 
        beat = list()

        for d in range(1, len(data)): # 0 index is Chart Head
            splitted = data[d].split(' ')
            splitted = filter(None, splitted)
            next(splitted) # Time... Clipping
            pos = int(next(splitted)) # Sample ID
            arrhythmia_type = next(splitted) # Type
            for i in range(len(classes)):
                if arrhythmia_type in classes[i]:
                    arrhythmia_index = i
                    count_classes[arrhythmia_index] += 1
                    if(window_size <= pos and pos < (len(signals) - window_size)):
                        beat = signals[pos-window_size:pos+window_size]     ## REPLACE WITH R-PEAK DETECTION
                        # Plot an example to a beat    
                        if r is 1 and not example_beat_printed: 
                            plt.title("A Beat from " + records[1] + " Wave")
                            plt.plot(beat)
                            plt.show()
                            example_beat_printed = True

                        X.append(beat)
                        y.append(arrhythmia_index)

# data shape
print(np.shape(X), np.shape(y))

# COMMAND ----------

for i in range(0,len(X)):
        X[i] = np.append(X[i], y[i])

print(np.shape(X))

# COMMAND ----------

X_train_df = pd.DataFrame(X)
per_class = X_train_df[X_train_df.shape[1]-1].value_counts().sort_index()
print(per_class)
plt.figure(figsize=(20,10))
my_circle=plt.Circle( (0,0), 0.7, color='white')
#classes = ['N', 'F', 'Q', 'A', 'V']
plt.pie(per_class, labels=['Normal beat (N)', 'SuperaVentricular ectopic beat (SVEB)', 'Ventricular Ectopic beat (VEB)', 'Fusion of ventricular and normal beat (F)', 'Unknown Beat (Q)'], colors=['tab:blue','tab:orange','tab:purple','tab:olive','tab:green'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

# COMMAND ----------

df_1=X_train_df[X_train_df[X_train_df.shape[1]-1]==1]
df_2=X_train_df[X_train_df[X_train_df.shape[1]-1]==2]
df_3=X_train_df[X_train_df[X_train_df.shape[1]-1]==3]
df_4=X_train_df[X_train_df[X_train_df.shape[1]-1]==4]
df_5=X_train_df[X_train_df[X_train_df.shape[1]-1]==0]

# df_1_upsample=resample(df_1,replace=True,n_samples=10000,random_state=122)
# df_2_upsample=resample(df_2,replace=True,n_samples=10000,random_state=123)
# df_3_upsample=resample(df_3,replace=True,n_samples=5000,random_state=124)
# df_4_upsample=resample(df_4,replace=True,n_samples=10000,random_state=125)
df_5_downSample=resample(df_5, replace=False, n_samples=8000, random_state=126)

# X_train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample,df_5_upsample])
X_train_df=pd.concat([df_1,df_2,df_3,df_4,df_5_downSample])

# COMMAND ----------

sparkDF = sql_context.createDataFrame(X_train_df)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

assemblerInput = [str(x) for x in range(0, 280, 1)]

# COMMAND ----------

vector_assembler = VectorAssembler(
    inputCols=assemblerInput, outputCol='features'
)

# COMMAND ----------

sparkDFWithFeatures = vector_assembler.transform(sparkDF)

# COMMAND ----------

modelInputData = sparkDFWithFeatures.select('280', 'features').withColumnRenamed('280', 'label')

# COMMAND ----------

modelInputData.groupBy('label').count().show()
modelInputData.printSchema()

# COMMAND ----------

import random
import numpy as np
from functools import reduce
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.functions import rand,col,when,concat,substring,lit,udf,lower,sum as ps_sum,count as ps_count,row_number
from pyspark.sql.window import *
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler,BucketedRandomProjectionLSH,VectorSlicer
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.sql.functions import array, create_map, struct


def smote(vectorized_sdf,smote_config, min_):
    '''
    contains logic to perform smote oversampling, given a spark df with 2 classes
    inputs:
    * vectorized_sdf: cat cols are already stringindexed, num cols are assembled into 'features' vector
      df target col should be 'label'
    * smote_config: config obj containing smote parameters
    output:
    * oversampled_df: spark df after smote oversampling
    '''
    dataInput_min = vectorized_sdf[vectorized_sdf['label'] == min_]
    
    # LSH, bucketed random projection
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes",seed=smote_config.seed, bucketLength=smote_config.bucketLength)
    # smote only applies on existing minority instances    
    model = brp.fit(dataInput_min)
    model.transform(dataInput_min)
    
    # here distance is calculated from brp's param inputCol
    self_join_w_distance = model.approxSimilarityJoin(dataInput_min, dataInput_min, float("inf"), distCol="EuclideanDistance")
    
    # remove self-comparison (distance 0)
    self_join_w_distance = self_join_w_distance.filter(self_join_w_distance.EuclideanDistance > 0)
    
    over_original_rows = Window.partitionBy("datasetA").orderBy("EuclideanDistance")
    
    self_similarity_df = self_join_w_distance.withColumn("r_num", F.row_number().over(over_original_rows))
    
    self_similarity_df_selected = self_similarity_df.filter(self_similarity_df.r_num <= smote_config.k)

    over_original_rows_no_order = Window.partitionBy('datasetA')

    # list to store batches of synthetic data
    res = []
    
    # two udf for vector add and subtract, subtraction include a random factor [0,1]
    subtract_vector_udf = F.udf(lambda arr: random.uniform(0, 1)*(arr[0]-arr[1]), VectorUDT())
    add_vector_udf = F.udf(lambda arr: arr[0]+arr[1], VectorUDT())
    
    @udf(returnType=VectorUDT())
    def trimFeatures (arr):
        return Vectors.dense(arr.round(decimals=5))
    
    # retain original columns
    original_cols = dataInput_min.columns
    
    for i in range(smote_config.multiplier):
        print("generating batch %s of synthetic instances"%i)
        # logic to randomly select neighbour: pick the largest random number generated row as the neighbour
        df_random_sel = self_similarity_df_selected.withColumn("rand", F.rand()).withColumn('max_rand', F.max('rand').over(over_original_rows_no_order))\
                            .where(F.col('rand') == F.col('max_rand')).drop(*['max_rand','rand','r_num'])
        
        # create synthetic feature numerical part
        df_vec_diff = df_random_sel.select('*', subtract_vector_udf(F.array('datasetA.features', 'datasetB.features')).alias('vec_diff'))
        df_vec_modified = df_vec_diff.select('*', add_vector_udf(F.array('datasetA.features', 'vec_diff')).alias('features'))
        
        # for categorical cols, either pick original or the neighbour's cat values
        for c in original_cols:
            # randomly select neighbour or original data
            col_sub = random.choice(['datasetA','datasetB'])
            val = "{0}.{1}".format(col_sub,c)
            if c != 'features':
                # do not unpack original numerical features
                df_vec_modified = df_vec_modified.withColumn(c,F.col(val))
        
        # this df_vec_modified is the synthetic minority instances,
        df_vec_modified = df_vec_modified.drop(*['datasetA','datasetB','vec_diff','EuclideanDistance'])
        
        res.append(df_vec_modified)
    
    dfunion = reduce(DataFrame.unionAll, res)
    # union synthetic instances with original full (both minority and majority) df
    oversampled_df = dfunion.union(vectorized_sdf.select(dfunion.columns))
    #oversampled_df_ = oversampled_df.withColumn('features_', trimFeatures('features'))
    oversampled_df_ = oversampled_df.select(col('label'), trimFeatures(col('features')).alias("features"))
    return oversampled_df_

# COMMAND ----------

class smoteConfig:
    def __init__ (self, seed=1, bucketLength = 80, k=10, multiplier=12):
        self.seed = seed
        self.bucketLength = bucketLength
        self.k = k
        self.multiplier = multiplier
sC = smoteConfig()
df1 = smote (modelInputData, sC, 4.0)

# COMMAND ----------

sC = smoteConfig(1, 64, 4, 2)
df2 = smote (df1, sC, 3.0)

# COMMAND ----------

df2.groupBy('label').count().show()

# COMMAND ----------

# for SMOTE
X_train_df = df2

# COMMAND ----------

# for without SMOTE
X_train_df = modelInputData

# COMMAND ----------

plot = X_train_df.toPandas()
per_class = plot['label'].value_counts().sort_index()
data = [[0, "Normal beat (N)", per_class[0], 'blue'], [1, 'SuperaVentricular ectopic beat (SVEB)', per_class[1], 'orange'], [2, 'Ventricular Ectopic beat (VEB)', per_class[2], 'purple'], \
            [3, "Fusion of ventricular and normal beat (F)", per_class[3], 'olive'], [4, "Unknown Beat (Q)", per_class[4], 'green']]
data = pd.DataFrame(data, columns=['id', 'Class Label', 'count', 'color'])
print(data)
plt.figure(figsize=(20,10))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(per_class, labels=['Normal beat (N)', 'SuperaVentricular ectopic beat (SVEB)', 'Ventricular Ectopic beat (VEB)', 'Fusion of ventricular and normal beat (F)', 'Unknown Beat (Q)'], colors=['tab:blue','tab:orange','tab:purple','tab:olive','tab:green'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

# COMMAND ----------

train, test = X_train_df.randomSplit([0.8, 0.2], seed = 7)
test.printSchema()

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier


# COMMAND ----------

#lr = LogisticRegression(maxIter=50, regParam=0.01, elasticNetParam=0.0)
#dt = DecisionTreeClassifier(maxDepth=5)
#rf = RandomForestClassifier()
mpc = MultilayerPerceptronClassifier(maxIter=100, blockSize=64, layers=[280, 140, 5], seed=123)

# COMMAND ----------

# lrModel = lr.fit(train)
# dtModel = dt.fit(train)
# rfModel = rf.fit(train)

# COMMAND ----------

mpcModel = mpc.fit(train)

# COMMAND ----------

mpcModel.save("file:///media/nikhil/Nikhil/mpc_score_model_SMOTE")

# COMMAND ----------

mpcModel.save("file:///media/nikhil/Nikhil/mpc_score_model_WITHOUT_SMOTE")

# COMMAND ----------

# lrModel.save("lr_score_model")
# dtModel.save("dt_score_model")
# rfModel.save("rf_score_model")

# COMMAND ----------

# lrPrediction = lrModel.transform(test)
# dtPrediction = dtModel.transform(test)
# rfPrediction = rfModel.transform(test)

# COMMAND ----------

mpcPrediction = mpcModel.transform(test)

# COMMAND ----------

# evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
# evaluator.evaluate(lrPrediction)
# evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
# evaluator.evaluate(dtPrediction)
# evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
# evaluator.evaluate(rfPrediction)

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassificationModel

mpcSMOTE_Model = MultilayerPerceptronClassificationModel.load("file:///media/nikhil/Nikhil/mpc_score_model_SMOTE")


# COMMAND ----------

mpcWithoutSOME_Model = MultilayerPerceptronClassificationModel.load("file:///media/nikhil/Nikhil/mpc_score_model_WITHOUT_SMOTE")

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

mpcPrediction = mpcSMOTE_Model.transform(test)

evaluator = MulticlassClassificationEvaluator()
evaluator.setLabelCol("label")
evaluator.setPredictionCol ("prediction")
evaluator.evaluate(mpcPrediction)
print("Test Area under ROC (SMOTE): ", evaluator.evaluate (mpcPrediction))

preds_and_labels = mpcPrediction.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
# Weighted stats
print ("Normal beat (N)")
print("recall = %s" % metrics.recall(0))
print("precision = %s" % metrics.precision(0))

print ("SuperaVentricular ectopic beat (SVEB)")
print("recall = %s" % metrics.recall(1))
print("precision = %s" % metrics.precision(1))

print ("Ventricular Ectopic beat (VEB)")
print("recall = %s" % metrics.recall(2))
print("precision = %s" % metrics.precision(2))

print ("Fusion of ventricular and normal beat (F)")
print("recall = %s" % metrics.recall(3))
print("precision = %s" % metrics.precision(3))

print ("Unknown Beat (Q)")
print("recall = %s" % metrics.recall(4))
print("precision = %s" % metrics.precision(4))
#print("Weighted F(1) Score = %s" % metrics.fMeasure(1))
#print("Weighted F(0.5) Score = %s" % metrics.fMeasure(0, beta=0.5))
#print("Weighted false positive rate = %s" % metrics.falsePositiveRate(0))

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

mpcPrediction = mpcWithoutSOME_Model.transform(test)

evaluator = MulticlassClassificationEvaluator()
evaluator.setLabelCol("label")
evaluator.setPredictionCol ("prediction")
evaluator.evaluate(mpcPrediction)
print("Test Area under ROC (Without SMOTE): ", evaluator.evaluate (mpcPrediction))

preds_and_labels = mpcPrediction.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
# Weighted stats
print ("Normal beat (N)")
print("recall = %s" % metrics.recall(0))
print("precision = %s" % metrics.precision(0))

print ("SuperaVentricular ectopic beat (SVEB)")
print("recall = %s" % metrics.recall(1))
print("precision = %s" % metrics.precision(1))

print ("Ventricular Ectopic beat (VEB)")
print("recall = %s" % metrics.recall(2))
print("precision = %s" % metrics.precision(2))

print ("Fusion of ventricular and normal beat (F)")
print("recall = %s" % metrics.recall(3))
print("precision = %s" % metrics.precision(3))

print ("Unknown Beat (Q)")
print("recall = %s" % metrics.recall(4))
print("precision = %s" % metrics.precision(4))
#print("Weighted F(1) Score = %s" % metrics.fMeasure(1))
#print("Weighted F(0.5) Score = %s" % metrics.fMeasure(0, beta=0.5))
#print("Weighted false positive rate = %s" % metrics.falsePositiveRate(0))

# COMMAND ----------

def incrementalLearning (weights):
    mpc = MultilayerPerceptronClassifier(maxIter=100, blockSize=64, initialWeights=weights, layers=[280, 140, 5], seed=123)

# COMMAND ----------

weights = mpcModel.weights
incrementalLearning (weights)
