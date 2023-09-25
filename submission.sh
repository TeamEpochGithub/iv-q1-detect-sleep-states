#!/bin/bash

# Remove everything in old submission_zip folder
rm -rf submission_zip
mkdir submission_zip

# Copy metadata file to submission_zip folder
cp dataset-metadata.json submission_zip/

# Zip files in source folder and copy python files
zip -j -r submission_zip/cv.zip source/cv/
zip -j -r submission_zip/ensemble.zip source/ensemble/
zip -j -r submission_zip/feature_engineering.zip source/feature_engineering/
zip -j -r submission_zip/hpo.zip source/hpo/
zip -j -r submission_zip/loss.zip source/loss/
zip -j -r submission_zip/models.zip source/models/
zip -j -r submission_zip/preprocessing.zip source/preprocessing/
zip -j -r submission_zip/score.zip source/score/
zip -j -r submission_zip/configs source/configs/
cp source/submit_to_kaggle.py submission_zip/
cp source/config.json submission_zip/

# Submit folder to kaggle as new dataset version
kaggle datasets version -p submission_zip -m "Update submission"