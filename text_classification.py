# !/usr/bin/python3
# -*- coding: utf-8 -*-


"""test.py: test file."""

__author__		= 'Ritesh Dubal'


import os
import sys
import json
import time
import pandas as pd
import random
import multiprocessing
import csv
import nltk
import boto3
import argparse
import tarfile
import sagemaker
try:
	nltk.data.find('punkt.zip')
except:
	nltk.download('punkt')


def read_json_file(filePath):
	file = open(filePath)
	data = json.loads(file.read())
	file.close()
	return data

config = read_json_file('configuration.json')

_s3_bucket_name = config['_s3_bucket_name']
_s3_access_key = config['_s3_access_key']
_s3_secret_key = config['_s3_secret_key']
_s3_region_name = config['_s3_region_name']

_s3_raw_dataset_dir = '001_raw_dataset/'
_s3_dataset_dir = '002_dataset/'
_s3_preprocessed_dataset_dir = '003_dataset/'
_s3_output_dir = '004_output/'

_dataset = 'dbpedia_csv.tar.gz'
_dataset_dir = 'dbpedia_csv/'

_train_dataset_filePath = 'dbpedia_csv/train.csv'
_test_dataset_filePath = 'dbpedia_csv/test.csv'
_classes_filePath = 'dbpedia_csv/classes.txt'

_transformed_train_filePath = 'dbpedia.train'
_transformed_test_filePath = 'dbpedia.validation'


s3_obj = boto3.resource("s3", region_name=_s3_region_name, aws_access_key_id=_s3_access_key, aws_secret_access_key=_s3_secret_key)
bucket_name = s3_obj.Bucket(_s3_bucket_name)


############################################################
# Downloading from S3 to local
############################################################
def download_and_extract_from_s3_to_local():
	print()
	print("Downloading dataset")
	dataset_path = _s3_raw_dataset_dir + _dataset
	bucket_name.download_file(dataset_path, _dataset)
	print("Dataset downloaded")
	tar = tarfile.open(_dataset, "r:gz")
	tar.extractall()
	tar.close()
	print("Dataset extracted")

def upload_from_local_to_s3(source, destination):
	bucket_name.upload_file(source, destination)

def upload_extracted_dataset_to_s3():
	print()
	print("uploading dataset to s3")
	for root, dirs, files in os.walk(_dataset_dir):
		for filename in files:
			local_path = os.path.join(root, filename)
			upload_from_local_to_s3(local_path, _s3_dataset_dir + local_path)
	print("Dataset uploaded to s3")


############################################################
# Transform the dataset and Preprocess for NLP model
############################################################
index_to_label = {}
if os.path.exists(_classes_filePath):
	with open(_classes_filePath) as f:
		for i,label in enumerate(f.readlines()):
			index_to_label[str(i+1)] = label.strip()

def transform_instance(row):
	cur_row = []
	label = "__label__" + index_to_label[row[0]]
	cur_row.append(label)
	cur_row.extend(nltk.word_tokenize(row[1].lower()))
	cur_row.extend(nltk.word_tokenize(row[2].lower()))
	return cur_row

def transform_and_preprocess():
	def preprocess(input_file, output_file, keep=1):
		all_rows = []
		with open(input_file, 'r') as csvinfile:
			csv_reader = csv.reader(csvinfile, delimiter=',')
			for row in csv_reader:
				all_rows.append(row)
		random.shuffle(all_rows)
		all_rows = all_rows[:int(keep*len(all_rows))]
		pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
		transformed_rows = pool.map(transform_instance, all_rows)
		pool.close() 
		pool.join()
		with open(output_file, 'w') as csvoutfile:
			csv_writer = csv.writer(csvoutfile, delimiter=' ', lineterminator='\n') # notice the delimiter.
			csv_writer.writerows(transformed_rows)

	print()
	print("Preprocessing and transforming train and validation dataset")
	preprocess(_train_dataset_filePath, _transformed_train_filePath, keep=.2)
	preprocess(_test_dataset_filePath, _transformed_test_filePath)
	print("Preprocessing and transforming train and validation dataset completed")

	upload_from_local_to_s3(_transformed_train_filePath, _s3_preprocessed_dataset_dir+_transformed_train_filePath)
	upload_from_local_to_s3(_transformed_test_filePath, _s3_preprocessed_dataset_dir+_transformed_test_filePath)
	print("Preprocessing and transforming train and validation dataset uploaded to s3")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='A program to define text classification modes.')
	parser.add_argument('-m', '--mode', type=str, default='test', help='Define mode')
	args = parser.parse_args()
	mode = args.mode

	if mode == 'download':
		download_and_extract_from_s3_to_local() # Download the dataset from s3
	elif mode == 'upload':
		upload_extracted_dataset_to_s3() # Upload the dataset to s3
	elif mode == 'transform_and_preprocess':
		transform_and_preprocess() # Upload the transformed and preprocessed data to s3
	else:
		print("Invalid argument passed!")