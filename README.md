# Amazon Employee Access Challenge (Kaggle)



## Introduction

The problem statement is taken from Kaggle competition's Amazon Employee Access Challenge. When a new employee joins the company he or needs a variety of access to systems and portals at different levels depending on the designation, business unit, role etc of the employee.
There is a considerable amount of data regarding an employee’s role within an organization and the resources to which they have access. The data consists of real historical data collected from 2010 & 2011. Employees are manually allowed or denied access to resources over time.
The competition provides two files:
    1) train.csv - The training set. Each row has the ACTION (ground truth), RESOURCE, and
information about the employee's role at the time of approval
    2) test.csv - The test set for which predictions should be made. Each row asks whether an
employee having the listed characteristics should have access to the listed resource.

The actual competition could be found at: https://www.kaggle.com/competitions/amazon-employee-access-challenge/overview

## Visualizing  raw and augmented data:

<img width="559" alt="image" src="https://user-images.githubusercontent.com/55175448/175791298-eb7c86da-df64-4954-9e12-5322086203f5.png">

##
We created three machine learning models with different attributes and uploaded our result on Kaggle to check our model accruacies:

### 1) Shallow Model (RandomForestClassifier)

<img width="573" alt="image" src="https://user-images.githubusercontent.com/55175448/175791483-7227381c-ded0-4e47-8aae-aa78b7a059ba.png">


### 2) 3 Layer Neural Network Model

### 3) Deep (5 Layer) Neural Network Model

###

Detailed project report can be found at: [Group2_Homework6_FinalReport.pdf](https://github.com/Atharva4320/Amazon-Employee-Access-Challenge/files/8985622/Group2_Homework6_FinalReport.pdf)
