import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV ,ShuffleSplit 
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score 
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from imblearn.over_sampling import RandomOverSampler,SMOTE,SMOTENC,ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier #import DecisionTreeClassifier function from scikit-learn
from sklearn.neighbors import KNeighborsClassifier #import KNeighborsClassifier function from scikit-learn
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier function from scikit-learn
from sklearn.ensemble import GradientBoostingClassifier  # Import GradientBoostingClassifier function from scikit-learn
from yellowbrick.classifier import ClassificationReport #import ClassificationReport function from yellowbrick library
from xgboost import XGBClassifier  # Import XGBoost function from xgboost library
from sklearn.metrics import roc_auc_score,roc_curve
import pickle #import pickle library to serialize and deserialize Python objects
import warnings #import warnings library to handle warnings
import pywebio  # Web app tool for developing end user apps
from pywebio import * #import all functions from pywebio library
from pywebio.input import * #import input functions from pywebio library
from pywebio.output import * #import output functions from pywebio library
from pprint import pprint  # Pretty print library
import random as rd #import random library as rd
import numpy as np  # Import numpy for numeric calculations
import pandas as pd  # Import pandas for dataframe analysis
import seaborn as sns  # Import seaborn for visualization
from collections import Counter #import Counter function from collections library
from scipy.stats import randint, uniform #import randint and uniform functions from scipy.stats library
import matplotlib.pyplot as plt  # Import matplotlib for visualization
from matplotlib.pyplot import Figure #import Figure function from matplotlib.pyplot library
from sklearn import preprocessing  # Import preprocessing function from scikit-learn
from imblearn.over_sampling import SMOTE  # Import SMOTE function from imblearn
from sklearn.model_selection import train_test_split  # Import train_test_split function from scikit-learn
from imblearn.under_sampling import RandomUnderSampler  # Import RandomUnderSampler function from imblearn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler  # Import various preprocessing functions from scikit-learn
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression function from scikit-learn
from sklearn.feature_selection import RFE  # Import Recursive Feature Elimination function from scikit-learn
from sklearn.tree import DecisionTreeClassifier #import DecisionTreeClassifier function from scikit-learn
from sklearn.neighbors import KNeighborsClassifier #import KNeighborsClassifier function from scikit-learn
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier function from scikit-learn
from sklearn.ensemble import GradientBoostingClassifier  # Import GradientBoostingClassifier function from scikit-learn
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV  
from yellowbrick.classifier import ClassificationReport #import ClassificationReport function from yellowbrick library
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score  
from sklearn.svm import SVC  # Import Support Vector Machine function from scikit-learn
from xgboost import XGBClassifier  # Import XGBoost function from xgboost library
from mpl_toolkits import mplot3d  # Import 3D plotting function from matplotlib
from sklearn.utils import class_weight  # Import class_weight function from scikit-learn
import warnings
warnings.filterwarnings('ignore')
from aequitas.group import Group 

class modeling:
    
    def __init__(self,X_train,X_test,y_train,y_test,group_train,group_test,best=None,name=None):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.model=None
        self.grid=None
        self.best=best
        self.group_train=group_train
        self.group_test=group_test
        self.name=name
        
    def get_fairness_metrics(self,groups,y_true, y_pred,FIXED_FPR=0.05):
        labels = self.y_test
        groups = self.group_test
        g = Group()
        aequitas_df = pd.DataFrame(
            {"score": y_pred,
             "label_value": y_true,
             "group": groups}
        )
        # Use aequitas to compute confusion matrix metrics for every group.
        disparities_df = g.get_crosstabs(aequitas_df, score_thresholds={"score_val": [FIXED_FPR]})[0]

        # Predictive equality is the differences in FPR (we use ratios in the paper)
        predictive_equality = disparities_df["fpr"].min() / disparities_df["fpr"].max()

        return predictive_equality, disparities_df
        
    def train(self, model, param_grid, search_type='grid'):
        
        self.model = model
        if search_type == 'grid':
            search = GridSearchCV(self.model, param_grid,cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),scoring='roc_auc',verbose=2,n_jobs=-1)
        elif search_type == 'random':
            search = RandomizedSearchCV(self.model, param_grid,scoring='roc_auc',cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),n_iter=20,verbose=2,n_jobs=-1)
        elif search_type == 'bayesian':
            search = BayesSearchCV(self.model, param_grid,  scoring='roc_auc')
        else:
            raise ValueError("Invalid search_type. Supported types are 'grid', 'random', and 'bayesian'.")
        
        search.fit(self.X_train, self.y_train)
        print('Best hyperparameters:', search.best_params_)
        print('Best estimator:', search.best_estimator_)
        print('Best score:', search.best_score_)
        self.best = search.best_estimator_
        # Evaluating fairness metrics and plotting ROC curve with accuracy
        self.plot_auc_accuracy()
        if self.name !="catboost":
        # Plotting confusion matrix with classification report
            self.plot_confmat_classreport()
        else:
            y_pred = self.best.predict(self.X_test)
            print(classification_report(self.y_test,y_pred))
            
    
    def plot_confmat_classreport(self):
        # Calculating Prediction
        y_pred = self.best.predict(self.X_test)

        # Calculating Confusion Matrix and plotting it
        CM = confusion_matrix(self.y_test, y_pred)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        sns.heatmap(CM, annot=True, fmt='g', cmap='Pastel1', ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Truth')
        axes[0].set_title('Confusion Matrix')

        # Calculating Classification Report and plotting it
        classes = ['0', '1']
        visualizer = ClassificationReport(self.best, classes=classes, support=True, ax=axes[1])
        visualizer.fit(self.X_train, self.y_train)
        visualizer.score(self.X_test, self.y_test)
        visualizer.show()
        axes[1].set_title('Classification Report')

        # Adjusting the layout of the subplots and displaying the plot
        fig.tight_layout()
        plt.show()

    def plot_auc_accuracy(self, FIXED_FPR=0.05):
        # Calculating AUC and plotting the ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.best.predict_proba(self.X_test)[:, 1])
        roc_auc = roc_auc_score(self.y_test, self.best.predict_proba(self.X_test)[:, 1])
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        axes[0].plot(fpr, tpr, color='purple', label='AUC = %0.2f' % roc_auc)
        axes[0].plot([0, 1], [0, 1], color='magenta', linestyle='--')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('Receiver Operating Characteristic (ROC) Curve')

        # Calculating Accuracy and plotting it
        tpr_fixed_fpr = tpr[fpr < FIXED_FPR][-1]
        fpr_fixed_fpr = fpr[fpr < FIXED_FPR][-1]
        threshold_fixed_fpr = thresholds[fpr < FIXED_FPR][-1]

        print("AUC:", roc_auc)
        to_pct = lambda x: str(round(x, 4) * 100) + "%"
        print("TPR: ", to_pct(tpr_fixed_fpr), "\nFPR: ", to_pct(fpr_fixed_fpr), "\nThreshold: ", round(threshold_fixed_fpr, 2))

        # Calling get_fairness_metrics to calculate predictive equality
        predictive_equality, disparities_df = self.get_fairness_metrics(self.group_test,self.y_test, self.best.predict_proba(self.X_test)[:, 1], FIXED_FPR)

        print("Predictive Equality: ", to_pct(predictive_equality))

        accuracy = self.best.score(self.X_test, self.y_test)
        axes[1].bar(['Accuracy'], [accuracy], color=['purple'])
        axes[1].set_ylim([0, 1.5])
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].set_xticks([])
        axes[1].tick_params(axis='y', labelsize=10)

        # Add accuracy value as text label on the accuracy bar plot
        axes[1].text(0, accuracy + 0.05, '{:.2f}'.format(accuracy), horizontalalignment='center', fontsize=15)

        # Adjusting the layout of the subplots and displaying the plot
        fig.tight_layout()
        plt.show()

    def save_model(self,filename):
        # Save the model to disk using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.best, f)
        
        # Return the path to the saved model file
        return filename
