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
  

# our repo contain four folders each for one step and the my_package folder which contain the classes used in the implementation:
1. Step 1 : which contain 7 notebooks one for each model hyperparameter tuning.
   as we have used the four baseline models and tried catboost,adaboost and lgbm.
## results of the first step

| Model     | AUC | TPR    | Predictive equality  |
| --------- | --- | ------------- |-----------|
| Logistic Regression     | 0.879 | 49.65%      |88.42%|
| Random Forest     | 0.872 | 48.68%     |96.23%|
| Neural Network  | 0.884  | 52.19%     | 99.25%|
| AdaBoost | 0.893  | 52.40%     |  100.0%|
| XGBoost | 0.886  |54.66%     |  88.81%|
| CatBoost | 0.895  |55.14%    |   86.27%|
| LGBM |0.886  |51.91%    |    79.99%|








