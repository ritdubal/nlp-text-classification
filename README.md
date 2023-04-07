# text_classification

Suppose you have been tasked with setting up a machine learning pipeline to deploy an NLP model in an AWS environment. The model needs to be trained on a large dataset of text files stored in an S3 bucket. The training data needs to be pre-processed and transformed before being fed into the model, and the output of the model needs to be stored in another S3 bucket. Write a Python script to accomplish the following tasks.

1. Download the training data from the S3 bucket to the local machine.
2. Preprocess the data by tokenizing the text and removing stop words and special characters.
3. Transform the preprocessed data into a format suitable for the NLP model.
4. Train the NLP model on the transformed data.
5. Evaluate the model's performance on a separate validation set.
6. Store the output of the model in the output S3 bucket.

The script should be designed to be run on an AWS EC2 instance and should make use of AWS services such as S3 and SageMaker. Please include comments in your code to explain your thought process and the reasoning behind your design decisions

## Environment - Local
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Dataset - dbpedia
DBpedia (from "DB" for "database") is a project aiming to extract structured content from the information created in Wikipedia.

## Problem Statement on NLP
From dataset predict the categorical classes based on input statement provided from wikipedia content.

## Approach
## Local Environment
1. Download the data locally and upload in s3 bucket
```
wget https://github.com/saurabh3949/Text-Classification-Datasets/raw/master/dbpedia_csv.tar.gz
aws s3 cp dbpedia_csv.tar.gz s3://sagemaker-ap-south-1-907831156916/001_raw_dataset/
```

2. To download the data from S3 and extract the dataset locally
```
python3 text_classification.py -m download
```

3. To upload extracted folder on S3
```
python3 text_classification.py -m download
```

4. To upload Transformed and Preprocessed trained and validation dataset on S3
```
python3 text_classification.py -m transform_and_preprocess
```

## AWS SageMaker Environment
5. Train the NLP train data on AWS SageMaker
6. Evaluate model's performance on validation set
7. Store the output on S3 Bucket

For Step 5,6,7 follow Jupyter Notebook code in AWS SageMaker.