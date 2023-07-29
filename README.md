# "The Bank Account Fraud Dataset: A Large-Scale Tabular Dataset for Evaluating Machine Learning Performance and Fairness"

## Introduction:
The paper addresses the lack of publicly available, large-scale, realistic tabular datasets for evaluating the performance and fairness of Machine Learning (ML) models in real-world applications. While there has been significant progress in the availability of unstructured data resources for computer vision and natural language processing (NLP) tasks, tabular data, which is prevalent in many high-stakes domains, has not received the same level of attention. Additionally, existing tabular datasets used in Fair ML literature suffer from various limitations, as detailed in Section 2 of the paper. To bridge this gap, the authors introduce the Bank Account Fraud (BAF) dataset, the first publicly available, privacy-preserving, large-scale, and realistic suite of tabular datasets.
so the dataset can be used in many problems as :
1. Fraud Detection:
   - Description: Fraud detection is the process of identifying and preventing fraudulent activities, such as unauthorized transactions, identity theft, and other deceptive behaviors in banking and financial systems.
   - How the Dataset Helps: The BAF dataset provides a realistic and diverse set of features related to banking transactions, account information, and customer attributes. ML models trained on this dataset can learn patterns indicative of fraud, enabling financial institutions to better detect and prevent fraudulent activities, thereby safeguarding their customers and assets.

2. Fairness Evaluation:
   - Description: Fairness evaluation in ML aims to assess and address potential biases that ML models may exhibit in decision-making processes. It ensures that the models' predictions are not systematically biased against particular demographic groups or protected characteristics (e.g., race, gender).
   - How the Dataset Helps: The BAF dataset is designed to be privacy-preserving and includes features that may introduce fairness concerns. By evaluating ML models on this dataset, researchers and practitioners can analyze whether the models demonstrate disparate treatment or impact on different groups, and work towards developing fairer algorithms.

3. Model Generalization:
   - Description: Model generalization refers to the ability of an ML model to perform well on unseen data from the same distribution as the training data. Generalization is crucial to ensure that ML models can handle real-world scenarios effectively.
   - How the Dataset Helps: With its large-scale and realistic nature, the BAF dataset offers an excellent opportunity to evaluate how well ML models generalize to new, unseen banking transactions and customer data. This analysis helps researchers identify potential overfitting or underfitting issues, improving the model's robustness.

4. Model Robustness:
   - Description: Model robustness involves testing ML models against various adversarial attacks or data perturbations to ensure their performance remains stable and reliable in the face of potential threats.
   - How the Dataset Helps: The BAF dataset can be used to assess how ML models handle adversarial attempts to deceive the system, such as injecting noise into the data or perturbing features. Evaluating model robustness on the BAF dataset helps improve the model's resilience and security against potential attacks.

5. Performance Evaluation:
   - Description: Performance evaluation involves measuring the effectiveness of ML models in making predictions or classifications. It helps researchers compare different models and select the most suitable one for a specific task.
   - How the Dataset Helps: With the BAF dataset providing ground truth labels for fraudulent transactions, researchers can evaluate model performance using metrics like accuracy, precision, recall, F1-score, and ROC curves. These evaluations aid in determining which models perform best in detecting fraud and producing reliable results.

##  Motivation:
The five problems mentioned are essential to solve due to their significant impact on various aspects of real-world applications and the overall success and reliability of machine learning models. Let's explore why each of these problems is important:

1. Fraud Detection:
   - Importance: Fraud detection is crucial in the banking and financial industry to protect customers and financial institutions from losses due to unauthorized and deceptive activities. The ability to accurately detect fraud ensures the integrity and security of financial transactions, maintains customer trust, and minimizes financial losses for both individuals and organizations.

2. Fairness Evaluation:
   - Importance: Fairness evaluation in machine learning is critical to ensuring that the decisions made by ML models do not systematically discriminate against specific demographic groups or protected attributes. Addressing fairness concerns is necessary to avoid biased decision-making, maintain ethical standards, and prevent potential harm or discrimination against certain groups of individuals.

3. Model Generalization:
   - Importance: Model generalization is fundamental to ensure that ML models can effectively handle real-world scenarios and unseen data. A model that generalizes well is more likely to provide accurate predictions when faced with new and previously unseen examples. Robust generalization is essential for deploying ML models in various applications, where the data distribution may change over time.

4. Model Robustness:
   - Importance: ML models can be susceptible to adversarial attacks, where malicious actors intentionally manipulate the data to deceive the model. Robustness testing helps identify vulnerabilities and ensures that the model's performance remains stable and reliable even in the presence of adversarial attempts, safeguarding against potential security breaches and ensuring trustworthy predictions.

5. Performance Evaluation:
   - Importance: Accurate performance evaluation is crucial for selecting the most effective ML model for a specific task. Properly evaluating model performance allows researchers and practitioners to compare different models objectively, choose the best-performing one, and identify areas for improvement. Reliable performance evaluation is the basis for making informed decisions about model deployment and optimization.

## Data Describtion:
Sure, let's describe all the dataset variants using examples from a fraud detection dataset:

1. Dataset Base:
   - Description: The base variant represents a balanced fraud detection dataset with an equal number of positive instances (fraudulent transactions) and negative instances (non-fraudulent transactions). The demographic attributes of the customers are also well-distributed across the dataset.
   - Example: The base variant contains 10,000 credit card transactions, where 5,000 are fraudulent (positive instances) and 5,000 are non-fraudulent (negative instances). The dataset includes various customer demographics such as age, income, and credit score, with an equal representation of different age groups and income ranges.

2. Dataset Variant I:
   - Description: Variant I introduces higher group size disparity in the fraud detection dataset. It focuses on evaluating how well ML models perform when certain demographic groups are underrepresented or overrepresented in the data.
   - Example: In variant I, out of the 10,000 credit card transactions, 7,000 transactions are from customers aged 25-35 (Group A), while only 3,000 transactions are from customers aged 36-45 (Group B). This disparity in group sizes may impact the model's ability to detect fraud accurately for both groups.

3. Dataset Variant II:
   - Description: Variant II introduces higher prevalence disparity in the fraud detection dataset. It aims to evaluate fairness by introducing scenarios where certain demographic groups have a higher prevalence of fraudulent transactions than others.
   - Example: In variant II, out of the 5,000 fraudulent transactions, 3,000 transactions are from customers with low credit scores (Group A), while only 2,000 transactions are from customers with high credit scores (Group B). This prevalence disparity can highlight bias in the model's predictions, favoring one group over the other.

4. Dataset Variant III:
   - Description: Variant III provides better separability for one of the demographic groups in the fraud detection dataset. It evaluates how well ML models perform when there is a clear distinction between certain groups' characteristics in the data.
   - Example: In variant III, for Group A (customers with high income), there are distinct patterns and features associated with fraudulent transactions that make them easily separable from non-fraudulent transactions. However, for Group B (customers with low income), the fraudulent transactions are more mixed with non-fraudulent transactions, making separability challenging.

5. Dataset Variant IV:
   - Description: Variant IV introduces higher prevalence disparity in the training dataset of the fraud detection task. It examines fairness challenges in scenarios where the training data itself is imbalanced with respect to certain demographic groups.
   - Example: In variant IV, during the model training, the data used to learn fraud patterns is biased towards Group A, which is overrepresented with fraudulent transactions. This imbalance can lead to a model that may perform well on detecting fraud for Group A but might struggle to generalize to detect fraud for Group B.

6. Dataset Variant V:
   - Description: Variant V provides better separability in the training dataset for one of the demographic groups in the fraud detection task. It evaluates how well ML models perform when the training data has clear boundaries for a specific group's features.
   - Example: In variant V, the training data contains highly distinct patterns and features for fraudulent transactions associated with Group A, making them easily separable. However, for Group B, the fraudulent transactions have less distinct features, leading to potential challenges in generalizing the model to detect fraud for Group B in real-world scenarios.
  
## The evaluation metrics :

1. Performance Metric: Recall @ 5% FPR (False Positive Rate)
   - Description: The performance metric focuses on evaluating the model's ability to correctly identify fraudulent transactions (True Positive Rate) while controlling the False Positive Rate at 5%. Since fraud is a highly imbalanced task with approximately 1% fraud labels, accuracy alone is not adequate. Therefore, the authors use Recall, which measures the proportion of actual fraud cases correctly identified by the model, ensuring that the model is not biased towards predicting non-fraudulent transactions to achieve high accuracy.

2. Fairness Metric: Predictive Equality (FPR Balance)
   - Description: The fairness metric, Predictive Equality, is used to assess the fairness of the model's predictions across protected groups in the dataset. The authors measure the difference (or ratio) in False Positive Rate (FPR) between different demographic groups, considering the thresholds that yield a global 5% False Positive Rate. The choice of this fairness metric is justified by the nature of the classification scenario: a false positive may result in denying a bank account to a legitimate applicant, which can have significant societal impacts, while a false negative mainly incurs losses to the bank company.
  

our repo contain four folders each for one step and the my_package folder which contain the classes used in the implementation:
# 1. Step 1 : which contain 7 notebooks one for each model hyperparameter tuning.
   as we have used the four baseline models and tried catboost,adaboost and lgbm.

## the baseline results before tuning:
-- the preprocessing was only using standard scaler for numerical features and one hot encoding for categorical features.
| Model     | AUC | TPR    | Predictive equality  |
| --------- | --- | ------------- |-----------|
| Logistic Regression     | 0.877 | 49.69%      |89.52%|
| Random Forest     |  0.805 |33.39%     | 34.14%|
| Neural Network  | 0.884  | 51.88%     |  84.36%|
| XGBoost | 0.867  |46.63%      | 76.07%|



## step 1 results after tuning the models and adding 3 different models

| Model     | AUC | TPR    | Predictive equality  |
| --------- | --- | ------------- |-----------|
| Logistic Regression     | 0.879 | 49.65%      |88.42%|
| Random Forest     | 0.872 | 48.68%     |96.23%|
| Neural Network  | 0.884  | 52.19%     | 99.25%|
| AdaBoost | 0.893  | 52.40%     |  100.0%|
| XGBoost | 0.886  |54.66%     |  88.81%|
| CatBoost | 0.895  |55.14%    |   86.27%|
| LGBM |0.886  |51.91%    |    79.99%|


### Baseline Results:
- The logistic regression model shows reasonable performance, with a good AUC and a moderate True Positive Rate (TPR). The fairness evaluation indicates good predictive equality, but there is still room for improvement in fairness across protected groups.
- The random forest model exhibits lower AUC and TPR compared to other models. Additionally, there are significant disparities in the False Positive Rate (FPR) across protected groups, highlighting the need for improved fairness.
- The neural network model demonstrates promising results with a high AUC and TPR. However, the fairness evaluation shows some disparities in the FPR across protected groups, indicating room for improvement in fairness.
- The XGBoost model performs reasonably well with a decent AUC and TPR. However, the fairness evaluation reveals disparities in the FPR between protected groups, suggesting possible fairness enhancements.

### Step 1 Results:
- After tuning, the logistic regression model's performance remains similar to the baseline, but there is a slight improvement in fairness.
- The random forest model shows significant improvement in both AUC and TPR compared to the baseline. The most notable enhancement is in predictive equality, indicating improved fairness.
- The neural network maintains its performance, but there are substantial improvements in fairness, with the predictive equality score significantly increasing.
- The AdaBoost model demonstrates impressive performance with a high AUC and TPR, achieving perfect fairness with a predictive equality score of 100.0%.
- the XGBoost model exhibits better performance and improved fairness compared to the baseline.
- The CatBoost model performs well, with good AUC and TPR, and it shows some improvement in fairness compared to the baseline models.
- The LGBM model maintains good  performance, and there is some enhancement in fairness, but further improvements are possible.

## comments on the first step:
The tuning and addition of new models have led to improvements in both performance and fairness metrics. The AdaBoost model stands out as achieving perfect fairness, but it's essential to carefully consider the trade-offs between performance and fairness when selecting the most suitable model for real-world applications.

# step 2 :
## in step 2 we have conducted four experiments with different preprocessing :
1.Experiment 1 : 
   - 1. we have replaced the -1 values to nan according to the data sheet.
   - 2. handle the missing values using mean and mode.
   - 3. detect and clip the outliers.
   - 4. use minmax scaler for numerical features and label encoding to categorical features.
   - 5. we have used the same hyperparameter earned from step 1 for each model.
   - 6. the results of this experiment are :
         | Model     | AUC | TPR    | Predictive equality  |
         | --------- | --- | ------------- |-----------|
         | Logistic Regression     | 0.867 | 46.11%     | 89.01%|
         | Random Forest     |  0.801 |  30.330%     |33.839%|
         | Neural Network  | 0.872  | 47.88%     |  93.63%|
         | AdaBoost |0.8893  |53.47%     |  100.0%|
         | XGBoost | 0.888  |52.99%     |  94.11%|
         | CatBoost(after tuning again) |0.893 | 54.93%      |  85.81%|
         | LGBM |0.882|50.38%     |    79.88%|

2.Experiment 2 : 
   - 1. we have replaced the -1 values to nan according to the data sheet.
   - 2. handle the missing values using mean and mode also deleted 'prev_address_months_count','bank_months_count'
   - 3. didn't handle the outliers.
   - 4. use robust scaler for numerical features and label encoding to categorical features.
   - 5. we have used the same hyperparameter earned from step 1 for each model.
   - 6. the results of this experiment are :
         | Model     | AUC | TPR    | Predictive equality  |
         | --------- | --- | ------------- |-----------|
         | Logistic Regression     | 0.859 |  44.89%     | 92.65%|
         | Random Forest     |  0.875 |   48.19%      |98.92%|
         | Neural Network  | 0.8749  | 48.26%     | 92.91%|
         | AdaBoost |0.885  |52.61%   |  100.0%|
         | XGBoost | 0.884  | 51.11%     | 80.259%|
         | CatBoost(after tuning again) |0.893 |55.35%        |   87.51%|
         | LGBM |0.882|50.42%     |    79.69%|


3.Experiment 3 : 
   - 1. we have replaced the -1 values to nan according to the data sheet.
   - 2. handle the missing values using mean and mode also deleted 'prev_address_months_count','bank_months_count'
   - 3. delete the outliers from the dataset.
   - 4. use standard  scaler for numerical features and one hot encoding encoding to categorical features.
   - 5. we have used the same hyperparameter earned from step 1 for each model.
   - 6. the results of this experiment are :
         | Model     | AUC | TPR    | Predictive equality  |
         | --------- | --- | ------------- |-----------|
         | Logistic Regression     | 0.877 | 48.55%      |  87.48%|
         | Random Forest     | 0.872 |   47.0%     |  99.53%|
         | Neural Network  | 0.878  | 49.81%      |  86.79%|
         | AdaBoost |0.884  |52.03%   |  100.0%|
         | XGBoost | 0.883  | 52.08%     |  91.38%|
         | CatBoost(after tuning again) |0.885 |51.94%         |  91.97%|
         | LGBM |0.871| 46.9%      |   82.56%|

4.Experiment 4 : 
   - 1. we have replaced the -1 values to nan according to the data sheet.
   - 2. handle the missing values using mean and mode also deleted 'prev_address_months_count'
   - 3. keep the outliers from the dataset.
   - 4. use robust  scaler for numerical features and one hot encoding encoding to categorical features.but delete the last column created for each feauture after one hot encoding manually
   - 5. we have used the same hyperparameter earned from step 1 for each model.
   - 6. the results of this experiment are :

      | Model              | AUC       | TPR     | Predictive Equality |
      | ------------------ | --------- | ------- | ------------------- |
      | Logistic Regression| 0.860     | 45.52%  | 100.00%             |
      | Random Forest      | 0.871     | 47.12%  | 100.00%             |
      | XGBoost            | 0.884     | 52.47%  | 100.00%             |
      | Deep Learning      | 0.884     | 51.39%  | 100.00%             |
      | AdaBoost           | 0.885     | 52.50%  | 100.00%             |
      | CatBoost           | 0.892     | 54.55%  | 100.00%             |
      | LGBM               | 0.876     | 49.44%  | 100.00%             |



