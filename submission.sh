#!/bin/bash

# Remove everything in old submission_zip folder
rm -rf submission_zip
mkdir submission_zip

# Copy metadata file to submission_zip folder
cp dataset-metadata.json submission_zip/

# Zip files in source folder and copy python files
zip -j -r submission_zip/cv.zip kaggle/input/source/cv/
zip -j -r submission_zip/ensemble.zip kaggle/input/source/ensemble/
zip -j -r submission_zip/feature_engineering.zip kaggle/input/source/feature_engineering/
zip -j -r submission_zip/hpo.zip kaggle/input/source/hpo/
zip -j -r submission_zip/loss.zip kaggle/input/source/loss/
zip -j -r submission_zip/models.zip kaggle/input/source/models/
zip -j -r submission_zip/preprocessing.zip kaggle/input/source/preprocessing/
zip -j -r submission_zip/score.zip kaggle/input/source/score/
zip -j -r submission_zip/configs kaggle/input/source/configs/
cp kaggle/input/source/submit_to_kaggle.py submission_zip/
cp kaggle/input/source/config.json submission_zip/
cp kaggle/input/source/__init__.py submission_zip/

# Submit folder to kaggle as new dataset version
kaggle datasets version -p submission_zip -m "Update submission"