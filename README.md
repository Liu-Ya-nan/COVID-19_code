# COVID-19_code

# Table of Contents
- [Project Title](#project-title)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)

# Installation

Install the f1.yml environment. 

# Usage

## Steps of runing code:
After installing the environment, proceed as follows：
1 - Train your network using "MODE = train" in main.py --> the trained model will be saved as Checkpoint in results folder.

Note: We train 5 differenct model where all have the same hyper parameter and set up but they have different random seed. Finally, for all the results presented in the paper we average the results of 5 run and report them. Thus for the results we want to report such as AUCs, FNRs, etc, when we are generating them it is essencial to remane them such that the name contain the run number. Later we use this naming protocol to gather and averaging the resilts over 5 run. At each section we wrote a guidline about what csv files needed to be raname and how.

2 - Test your network using "MODE = test" and runing main.py

The following csv files are generated: a) Eval.csv (contain AUC on validation set)  b)TestEval.csv (The AUC on test set for the model) c) True.csv (The true labels on Test set) d) preds.csv (The probability of each disease per image)  e) bipred.csv (The binary prediction of each label) f) Threshold.csv (The thereshold utilized to get binary predictions from probabilities per disease. It is calculated based on maximizing f1 score on validation set)

Rename TestEval.csv to Evel*.csv, where * is the number of run (e.g Evel1.csv for run1).

3 - Run preprocess() in FPRFNR. py to add metadata to the true labels of the test dataset and save it as True_withMeta.csv. This file and binary prediction bi_pred.csv of each result folder (associated to a random seed) are used to calculated TPRs.

4 - rename the results forlder followed by the applied random seed for the checkpoint. (e.g. for random seed 70 use results70)

Do the step 2 to 3 for all 5 runs per dataset.

5 - create a folder and call it "results" to save the results of combining the 5 run.

6 - Run the FPRFNR.py : Calculate the corresponding FPR/FNR value.

7 - Run the ConfidenceFromipynb.py : It gives: a) Percentage of images per attribiute in whole data (test, train and validation). b) AUC performance over 5 run.

8 - Run the plotFPRFNR.py : plot the disparity figures of the 5 run.
