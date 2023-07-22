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

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import ConfusionMatrixDisplay
# imports for neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

class nn_modeling:
    
    def __init__(self,model, X_train,X_test,y_train,y_test,group_train,group_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.group_train=group_train
        self.group_test=group_test
        self.model=model
        self.y_pred = None

    # Combine the compilation and training
    def compile_and_train(self, lr=1e-2):
        self.compile_model(lr)
        return self.train_model() 
    
    # compile a model using these specific metrics
    def compile_model(self, lr):
        metrics = [
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            self.f1,
        ]

        self.model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
            metrics=metrics
        )

    def train_model(self):
        # Use EarlyStopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            mode='max'
        )

        # Calculate the class wheights for the model, improves predictive equality
        class_weights = {0: 1., 1: np.sum(self.y_train == 0) / np.sum(self.y_train == 1)}

        hist = self.model.fit(
                        self.X_train, self.y_train,
                        class_weight=class_weights,batch_size=512,
                        epochs=100, # set lower if you only want to train for short period to get approximat results
                        callbacks=[early_stopping],
                        verbose=1,
                        validation_split=0.1 # Use 10% of training set as validation for EarlyStopping
        )
        # return the training history for possible visualization
        return hist

    # Evaluate a model by passing its output into the evaluate-function
    def score_keras_model(self):
        # Score the test set
        predictions = self.model.predict(self.X_test).flatten()
        self.y_pred = predictions
        self.plot_auc_accuracy_nn()

    def plot_auc_accuracy_nn(self, FIXED_FPR=0.05):
        # Calculating AUC and plotting the ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_pred)
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
        predictive_equality, disparities_df = self.get_fairness_metrics_nn(self.group_test,self.y_test,self.y_pred, FIXED_FPR)
        print("Predictive Equality: ", to_pct(predictive_equality))

        # extract the predicted class labels
        thresh = round(threshold_fixed_fpr, 2)
        predictions = np.where(self.y_pred > thresh, 1, 0) 

        accuracy = accuracy_score(self.y_test, predictions)
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

        #ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        self.plot_confmat_classreport_nn(predictions)
        print(classification_report(self.y_test, predictions))        


    def get_fairness_metrics_nn(self,groups,y_true, y_pred,FIXED_FPR=0.05):
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
        
    
    def plot_confmat_classreport_nn(self, predictions):
        
        # Calculating Confusion Matrix and plotting it
        CM = confusion_matrix(self.y_test, predictions)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        sns.heatmap(CM, annot=True, fmt='g', cmap='Pastel1', ax=axes)
        axes.set_xlabel('Predicted')
        axes.set_ylabel('Truth')
        axes.set_title('Confusion Matrix')

        # Adjusting the layout of the subplots and displaying the plot
        fig.tight_layout()
        plt.show()

    @keras.saving.register_keras_serializable(name="f1_func")
    def f1(self, y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    # --- Two currently unused metrics ---
    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def focal_loss(self, gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        return focal_loss_fixed

    def save_model(self,filename):
        # Save the model to disk
        self.model.save(filename+".keras")
        
        # Now, we can simply load without worrying about our custom objects or functions.
        # reconstructed_model = keras.models.load_model("filename.keras")

        # Return the path to the saved model file
        return filename
