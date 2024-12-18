# MushroomClassification

### Problem Statement
The Mushroom Classification problem involves identifying whether a mushroom is edible or poisonous based on various physical attributes, such as cap shape, color, gill size, and odor. The objective is to develop a model that can accurately classify mushrooms as either edible or poisonous, which is critical for preventing accidental consumption of harmful varieties.

This is a Binary Classification problem where the positive class indicates that the mushroom is poisonous, and the negative class indicates that the mushroom is safe to eat.

### Solution Proposed 
In this project, the primary focus is on accurately distinguishing edible mushrooms from poisonous ones by analyzing key features related to their physical attributes. The dataset's positive class corresponds to mushrooms identified as poisonous, while the negative class includes edible mushrooms.

In this project, logistic regression is used as the classification model to predict whether a mushroom is edible or poisonous based on its physical features. To address overfitting and improve model generalization, Principal Component Analysis (PCA) was applied to reduce the dimensionality of the data. By capturing the most informative components, PCA helps the model focus on key patterns, thus improving its performance and robustness in distinguishing between edible and poisonous mushrooms while reducing the risk of misclassification.

The goal is to minimize the risk of misclassification, particularly false negatives, as incorrectly identifying a poisonous mushroom as edible can lead to severe health risks. Therefore, the solution prioritizes reducing false predictions to ensure safe classification.
## Tech Stack Used
1. Python 
2. FastAPI 
3. Machine learning algorithms
4. Docker
5. MongoDB

## Infrastructure Required.

1. AWS S3
2. AWS EC2
3. AWS ECR
4. Git Actions
5. Terraform

## How to run?
Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for data storage. You also need AWS account to access the service like S3, ECR and EC2 instances.

## Data Collections
![image](https://user-images.githubusercontent.com/57321948/193536736-5ccff349-d1fb-486e-b920-02ad7974d089.png)


## Project Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536768-ae704adc-32d9-4c6c-b234-79c152f756c5.png)


## Deployment Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536973-4530fe7d-5509-4609-bfd2-cd702fc82423.png)