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
Note: AUC stands for Area Under the Curve, TPR stands for True Positive Rate (Sensitivity/Recall), FPR stands for False Positive Rate, and Predictive Equality is the measure of fairness. The higher the Predictive Equality score, the more equitable the model's predictions are across protected groups.


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


      | Model              | AUC       | TPR       | Predictive Equality |
      |--------------------|-----------|-----------|---------------------|
      | Logistic Regression| 0.8646    | 45.97%    | 90.09%              |
      | Random Forest      | 0.8724    | 48.05%    | 99.77%              |
      | XGBoost            | 0.8863    | 52.22%    | 82.13%              |
      | Deep Learning      | 0.8840    | 51.18%    | 93.76%              |
      | AdaBoost           | 0.8874    | 52.81%    | 100.0%              |
      | CatBoost           | 0.8933    | 54.86%    | 86.46%              |
      | LGBM               | 0.8789    | 49.79%    | 80.52%              |




# Step 3 handling the imbalanced problem:
## 1.Experiment 1:(using random undersampling)
Here are some reasons why one might use random undersampling on imbalanced data:

1. **Address Class Imbalance**: In imbalanced datasets, the majority class may dominate the learning process, leading to biased models that perform poorly on the minority class. By undersampling the majority class, you balance the class distribution, allowing the model to give more attention to the minority class during training.

2. **Reduce Overfitting**: When the class distribution is highly imbalanced, the classifier may become overfit to the majority class and struggle to generalize to unseen data. Random undersampling can reduce the risk of overfitting by providing a more balanced training set.

3. **Faster Training**: Imbalanced datasets with a large majority class can result in longer training times for the model. Random undersampling reduces the size of the training set, making the training process faster and more efficient.

However, random undersampling also has its drawbacks:

1. **Information Loss**: Random undersampling discards some samples from the majority class, potentially leading to the loss of valuable information present in those samples.

2. **Risk of Underfitting**: If the undersampling is too aggressive, it may remove essential information, resulting in underfitting, where the model fails to capture the underlying patterns in the data.

3. **Potential Bias**: Random undersampling can introduce bias by removing samples randomly from the majority class, potentially leading to a skewed representation of the data.

steps and result of experiment:   
   - 1. we have replaced the -1 values to nan according to the data sheet.
   - 2. handle the missing values using mean and mode also deleted 'prev_address_months_count'
   - 3. keep the outliers from the dataset.
   - 4. use robust  scaler for numerical features and one hot encoding encoding to categorical features.but delete the last column created for each feauture after one hot encoding manually
   - 5. we have used the same hyperparameter earned from step 1 for each model.
   - 6. the results of this experiment are:
        
      | Model                | AUC       | TPR      | Predictive Equality |
      |----------------------|-----------|----------|---------------------|
      | Logistic Regression  | 0.8697    | 47.12%   | 95.99%              |
      | Random Forest        | 0.8701    | 48.33%   | 99.96%              |
      | XGBoost              | 0.8858    | 52.08%   | 85.67%              |
      | Deep Learning        | 0.8819    | 50.49%   | 63.85%              |
      | AdaBoost             | 0.8827    | 51.77%   | 100.00%             |
      | CatBoost             | 0.8921    | 54.10%   | 89.29%              |
      | LGBM                 | 0.8889    | 52.74%   | 87.27%              |

## 2.Experiment 2:(using random oversampling):

Random Oversampling is one of the techniques used to address imbalanced data in a classification problem. It involves increasing the number of instances in the minority class by duplicating some of its samples randomly. The goal of random oversampling is to balance the class distribution and provide the classifier with enough examples from the minority class to learn better.

Here are some reasons why one might use random oversampling on imbalanced data:

1. **Improve Minority Class Representation**: In imbalanced datasets, the minority class may have very few samples, leading to poor representation and limited learning opportunities. By oversampling the minority class, you provide the classifier with more examples, making it easier for the model to learn patterns specific to that class.

2. **Address Class Imbalance**: Class imbalance can lead to biased models that favor the majority class. Random oversampling helps mitigate this issue by increasing the number of samples in the minority class, making the class distribution more balanced.

3. **Prevent Overfitting on Majority Class**: When the class distribution is highly imbalanced, the classifier may be biased towards the majority class, resulting in overfitting. Random oversampling helps prevent overfitting by increasing the representation of the minority class and reducing the chances of the model focusing solely on the majority class.

4. **Avoid Information Loss**: Other techniques like undersampling or data augmentation may discard valuable information from the majority class. Random oversampling preserves all the samples from the majority class while creating synthetic samples for the minority class.

However, it's essential to be cautious when using random oversampling, as it can lead to overfitting on the training set. If the random samples from the minority class are too similar to each other, the model may become overly confident in its predictions and struggle to generalize to unseen data.

steps and result of experiment:   
   - 1. we have replaced the -1 values to nan according to the data sheet.
   - 2. handle the missing values using mean and mode also deleted 'prev_address_months_count'
   - 3. keep the outliers from the dataset.
   - 4. use robust  scaler for numerical features and one hot encoding encoding to categorical features.but delete the last column created for each feauture after one hot encoding manually
   - 5. we have used the same hyperparameter earned from step 1 for each model.
   - 6. the results of this experiment are:

         | Model            | AUC      | TPR     | Predictive Equality |
         |------------------|----------|---------|---------------------|
         | Logistic Regression | 0.8652 | 46.35% | 92.96%             |
         | Random Forest       | 0.8724 | 48.12% | 99.79%             |
         | XGBoost             | 0.8863 | 53.27% | 88.72%             |
         | Deep Learning       | 0.8845 | 53.27% | 78.64%             |
         | AdaBoost            | 0.8874 | 53.02% | 100.00%            |
         | CatBoost            | 0.8937 | 54.76% | 85.90%             |
         | LGBM                | 0.8916 | 54.03% | 83.86%             |

## 3.Experiment 3 (using smote -nc variant):
SMOTE-NC (SMOTE for Nominal and Continuous features) is a variant of the Synthetic Minority Over-sampling Technique (SMOTE) that is specifically designed to handle datasets with both numerical (continuous) and categorical (nominal) features. It extends the original SMOTE algorithm to create synthetic samples for imbalanced datasets containing mixed data types.

The reasons to use SMOTE-NC on imbalanced data are as follows:

1. **Preserving Data Structure**: In real-world datasets, it is common to have a mix of numerical and categorical features. Using traditional SMOTE on such datasets may not consider the unique characteristics of categorical features, leading to synthetic samples that do not represent the original data's structure accurately. SMOTE-NC addresses this issue by handling both numerical and categorical features appropriately, preserving the data structure.

2. **Addressing Class Imbalance**: Imbalanced datasets have significantly fewer instances of the minority class compared to the majority class. This imbalance can lead to biased models that favor the majority class. SMOTE-NC helps alleviate class imbalance by generating synthetic samples for the minority class, increasing its representation in the dataset.

3. **Enhancing Generalization**: SMOTE-NC can improve the generalization ability of classifiers trained on imbalanced data. By generating synthetic samples, the classifier is exposed to more diverse instances of the minority class, reducing the risk of overfitting to the limited training data.

4. **Fairness Considerations**: When dealing with imbalanced data, fairness becomes a critical concern. SMOTE-NC can help improve fairness by providing a more balanced representation of different classes and potentially reducing the bias towards the majority class.

5. **Avoiding Data Loss**: Some resampling techniques, like random undersampling, remove instances from the majority class, resulting in data loss and a reduced training set size. SMOTE-NC, being an oversampling technique, retains all instances of the original data while creating synthetic samples for the minority class.

steps and results:
   - 1. we have replaced the -1 values to nan according to the data sheet.
   - 2. handle the missing values using mean and mode also deleted 'prev_address_months_count'
   - 3. keep the outliers from the dataset.
   - 4. use robust  scaler for numerical features and one hot encoding encoding to categorical features.but delete the last column created for each feauture after one hot encoding manually
   - 5. we have used the same hyperparameter earned from step 1 for each model.
   - 6. the results of this experiment are:
    
        
      | Model              | AUC     | TPR    | Predictive Equality |
      |--------------------|---------|--------|---------------------|
      | Logistic Regression| 0.8384  | 40.58% | 80.36%              |
      | Random Forest      | 0.8582  | 44.48% | 100.00%             |
      | XGBoost            | 0.8443  | 42.81% | 100.00%             |
      | Deep Learning      | 0.8393  | 38.60% | 61.05%              |
      | AdaBoost           | 0.8467  | 42.77% | 100.00%             |
      | CatBoost           | 0.8567  | 45.59% | 55.84%              |
      | LGBM               | 0.8462  | 43.71% | 65.23%              |



## 4.Experiment 4 (using undersampling then oversampling ):
Benefits of using undersampling_then_oversampling:

- Improved Generalization: By balancing the class distribution, the model becomes less biased towards the majority class and can learn patterns from the minority class more effectively, leading to improved generalization and better performance on unseen data.

- Reduced Overfitting: Addressing class imbalance with a balanced dataset can help reduce overfitting, as the model is less likely to memorize the majority class instances.

- Fairness Considerations: The resampling strategy aims to achieve fairness by providing an equal representation of different classes, reducing the risk of biased predictions towards the majority class.

- Computationally Efficient: Compared to some other resampling techniques, combining undersampling and oversampling is computationally less expensive, as it requires fewer synthetic samples.


steps and results:
   - 1. we have replaced the -1 values to nan according to the data sheet.
   - 2. handle the missing values using mean and mode also deleted 'prev_address_months_count'
   - 3. keep the outliers from the dataset.
   - 4. use robust  scaler for numerical features and one hot encoding encoding to categorical features.but delete the last column created for each feauture after one hot encoding manually
   - 5. we have used the same hyperparameter earned from step 1 for each model.
   - 6. the results of this experiment are:


         | Model              | AUC       | TPR       | Predictive Equality |
         |--------------------|-----------|-----------|---------------------|
         | Logistic Regression| 0.8509    | 43.29%    | 74.09%              |
         | Random Forest      | 0.8663    | 45.21%    | 90.91%              |
         | XGBoost            | 0.8742    | 49.76%    | 57.10%              |
         | Deep Learning      | 0.8648    | 45.21%    | 54.34%              |
         | AdaBoost           | 0.8727    | 48.85%    | 100.00%             |
         | CatBoost           | 0.8791    | 51.18%    | 60.85%              |
         | LGBM               | 0.8754    | 50.24%    | 54.95%              |

## 5.Experiment 5 (using ensemble models from imblearn ):
we found that there are some ensemble models that are designed specificaly to handle the imbalanced datasets automatically without the need for any additional data resampling techniques. These ensemble models can be found at "imblearn" library.
Below we list some of these models that we will try to assess in this notebook.
    - Bagging Algorithms
        - BalancedBaggingClassifier:  A Bagging classifier with additional balancing step to balance the training set at fit time.
        - BalancedRandomForestClassifier: A balanced random forest randomly under-samples each bootstrap sample to balance it.
    - Boosting Algorithms
        - RUSBoostClassifier: Randomly under-sample the dataset before performing a boosting iteration
        - EasyEnsembleClassifier: The classifier is an ensemble of AdaBoost learners trained on different balanced bootstrap samples. The balancing is achieved by random under-sampling.

n this trial of **Step 3** we performed the same pre-processing steps that were done during Step 2 for fair comparison. Pre-processing done is listed below:
* Nulls imputation
* Outliers deletion
* Scaling numerical features using "Standard Scaler"
* Encoding the categorical features using "One-hot-encoder"


**Thoughts on Training Results**

* During this experiment we tried multiple ensemble models that handle the class imbalance out-of-the-box without the need for manual data resampling techniques.

* BalancedBaggingClassifier
    *   Since we perfomred bagging of Logistic regression models we can compare this model performance with previously tried logistic regression models.
    *   We can notice that the AUC and TPR didn't change compared by the LR Basline and Step 1 and Step 2 models however the Predictive Equality (Fairness) has improved significantly from ~86% to ~95%.
    * so using the bagging with logistic regression has improved the LR performance as it kept the TPR and AUC without degradation and improved the fairness.


* BalancedRandomForestClassifier
    *   Since this ensemble model is just a balanced random forest which randomly under-samples each bootstrap sample to achieve balancing, we can compare its perfomance with the previous random forest models such as the Baseline, Step 1 and Step 2 models.
    * We can notice immediately that all the metrics have improved for the random forest when we performed balanced bagging where the fairness is improved tremendously from ~37% to 100%.
    * Also the TPR and AUC have been improved

* RUSBoostClassifier
    *   In RUSBoost model we used the default decision tree estimator with max depth of one so it resembles an AdaBoost model.
    * When we compare by AdaBoost we see that the fairness is preserved however the TPR has decreased while the AUC is approximately constant.


* EasyEnsembleClassifier
    *   “EasyEnsemble” is made specificaly from ensembled AdaBoost models combining the bagging and boosting and sampling.
    * So this model can be compared by AdaBoost models of Baseline, Step 1 and Step 2.
    * When we compare it by previous AdaBoost implementations we notice that the performance across all metrics is approximately constant suggesting that the base AdaBoost model is the best we can achieve using this model class as no improvement is noticed as with other models.
 


| Model                      | AUC       | TPR       | Predictive Equality |
|----------------------------|-----------|-----------|---------------------|
| Balanced Bagging Classifier| 0.8777    | 49.42%    | 95.37%              |
| Balanced Random Forest     | 0.8542    | 41.29%    | 100.00%             |
| RUS Boosting Classifier    | 0.8774    | 48.98%    | 100.00%             |
| Easy Ensemble Classifier   | 0.8861    | 52.08%    | 100.00%             |


## 6.Experiment 6 (using undersampling near miss then oversampling smote nc):
Benefits of using undersampling_then_oversampling:

Using NearMiss as an undersampling technique and SMOTE-NC (SMOTE for Nominal and Continuous features) as an oversampling technique in the same experiment can be a beneficial approach for dealing with imbalanced datasets.

1. **Addressing Class Imbalance**: NearMiss and SMOTE-NC both aim to address class imbalance, but they do it in different ways. NearMiss selects the majority class samples that are close to the minority class samples, effectively reducing the imbalance by removing instances. On the other hand, SMOTE-NC generates synthetic minority class samples by interpolating feature values, effectively increasing the number of instances in the minority class.

2. **Complementary Effects**: NearMiss focuses on preserving the boundary information between classes, which can be beneficial in scenarios where the classes have distinct clusters and are well-separated. It can help to improve the classifier's generalization by removing some redundant and noisy samples from the majority class.

3. **Handling Continuous and Categorical Features**: SMOTE-NC is specifically designed to handle datasets with both continuous and categorical features. It extends the original SMOTE algorithm to consider both types of features during the synthetic sample generation process, making it suitable for more diverse datasets.

4. **Better Generalization**: By combining NearMiss and SMOTE-NC, you can potentially create a more balanced dataset that retains essential information from both classes, leading to improved generalization of the classifier. It helps to create a more representative dataset that can better capture the underlying data distribution.

5. **Avoiding Overfitting**: Using both undersampling and oversampling techniques in combination can help avoid potential overfitting issues that may arise when using only one of the methods. By carefully balancing the class distribution, you reduce the chances of the model being biased towards either class.



steps and results:
   - 1. we have replaced the -1 values to nan according to the data sheet.
   - 2. handle the missing values using mean and mode also deleted 'prev_address_months_count'
   - 3. keep the outliers from the dataset.
   - 4. use robust  scaler for numerical features and one hot encoding encoding to categorical features.but delete the last column created for each feauture after one hot encoding manually
   - 5. we have used the same hyperparameter earned from step 1 for each model.
   - 6. the results of this experiment are:

      | Model             | AUC               | TPR     | Predictive Equality |
      |-------------------|-------------------|---------|---------------------|
      | Logistic Regression | 0.8946 | 57.44% | 68.26% |
      | Random Forest       | 0.9138 | 62.30% | 91.18% |
      | XGBoost             | 0.9372 | 73.31% | 57.96% |
      | Deep Learning       | 0.8648 | 45.21% | 54.34% |
      | AdaBoost            | 0.9100 | 63.17% | 100.00% |
      | CatBoost            | 0.9414 | 75.30% | 60.79% |
      | LGBM                | 0.9343 | 71.58% | 62.57% |


## 7.Experiment 7 (using undersampling near miss then oversampling smote nc and stratifying):

steps and results:
   - 1. we have replaced the -1 values to nan according to the data sheet.
   - 2. handle the missing values using mean and mode also deleted 'prev_address_months_count'
   - 3. keep the outliers from the dataset.
   - 4. split the data into train and test using stratification.
   - 5. use robust  scaler for numerical features and one hot encoding encoding to categorical features.but delete the last column created for each feauture after one hot encoding manually
   - 6. we have used the same hyperparameter earned from step 1 for each model.
   - 7. the results of this experiment are:


      | Model             | AUC               | TPR     | Predictive Equality |
      |-------------------|-------------------|---------|---------------------|
      | Logistic Regression | 0.8932 | 56.98% | 81.77% |
      | Random Forest       | 0.9210 | 65.50% | 94.18% |
      | XGBoost             | 0.9451 | 75.70% | 65.07% |
      | Deep Learning       | 0.9018 | 58.57% | 67.81% |
      | AdaBoost            | 0.9285 | 69.49% | 100.00% |
      | CatBoost            | 0.9546 | 79.69% | 55.26% |
      | LGBM                | 0.9489 | 77.70% | 56.87% |


# Step 4 : 
since the last experiment with nearmiss for undersampling followed by stratifying while creating the train and test sets then apply smote nc on the train data get the highest results.
we use the same technique with the preporcessing done in experiment 7 in step 3 on all the variants of the dataset and here are the reuslts.
- if we concern about the fairness and final output we could use adaboost and this is because of the following results:

| Model           | AUC              | TPR     | Predictive Equality |
|-----------------|------------------|---------|---------------------|
| Base AdaBoost   | 0.9346 | 72.08% | 100.0% |
| Variant 1 AdaBoost | 0.9324 | 69.63% | 100.0% |
| Variant 2 AdaBoost | 0.9378 | 71.21% | 100.0% |
| Variant 3 AdaBoost | 0.9329 | 70.67% | 100.0% |
| Variant 4 AdaBoost | 0.9383 | 72.21% | 100.0% |
| Variant 5 AdaBoost | 0.9310 | 69.58% | 100.0% |

The Predictive Equality score of 100.0% indicates that all variants achieve perfect fairness, with no disparity in the False Positive Rate (FPR) across protected groups. This means that all variants of the AdaBoost model are highly fair models.and for AUC the value slightly changes,indicating that the architecture could work on the five variants.

- but if we only concern on the fraud detection then we could use catboost which gives us those results:


| Model               | AUC              | TPR     | Predictive Equality |
|---------------------|------------------|---------|---------------------|
| CatBoost Base       | 0.9535 | 80.37% | 58.40% |
| CatBoost Variant 1  | 0.9538 | 77.88% | 98.03% |
| CatBoost Variant 2  | 0.9549 | 79.37% | 55.36% |
| CatBoost Variant 3  | 0.9534 | 78.47% | 92.92% |
| CatBoost Variant 4  | 0.9561 | 80.05% | 63.03% |
| CatBoost Variant 5  | 0.9520 | 78.15% | 87.26% |




Segregation of Duties:

| Duties               | Names          |
|------------------------|------------------|
| Classes                 |------------------|
| Data Cleaning           | Hassan Ahmed     |
| Data Preprocessing      | Amr Sayed        |
| Modeling                | Bilal Morsy      |
| NN Modeling             | Omar Amer        |
| Step 0                  |------------------|
| EDA                     | Belal Morsy      |
| Comparing Results       | Amr Sayed        |
| Step 1                  |------------------|
| LGBM & Random Forest    | Amr Sayed        |
| XGBoost and AdaBoost    | Hassan Ahmed     |
| Logistic Regression and Neural Networks | Omar Amer |
| Catboost                | Bilal Morsy      |
| Step 2                  |------------------|
| Pipeline 1              | Amr Sayed        |
| Pipeline 2              | Bilal Morsy      |
| Pipeline 3              | Omar Amer        |
| Pipeline 4              | Hassan Ahmed     |
| Step 3                  |------------------|
| Under Sampling         | Amr Sayed        |
| Over Sampling          | Amr Sayed        |
| Under Sampling then Over Sampling | Bilal Morsy |
| SMOTE - NC             | Bilal Morsy      |
| Imblearn ensemble       | Omar Amer        |
| Near miss and SMOTE - NC | Hassan Ahmed    |
| Near miss and SMOTE – NC and Stratifying | Omar Amer |
| Step 4                  |------------------|
| (CatBoost and AdaBoost and comparison) | Hassan Ahmed |
| Readme                  |------------------|
| Readme                  | Amr Sayed  & Hassan Ahmed     |
















