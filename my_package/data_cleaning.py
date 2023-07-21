import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import missingno as msno
from scipy import stats
import association_metrics as am
from scipy.stats import shapiro
class data_cleaning:
    
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.data.drop('device_fraud_count',axis=1,inplace=True)
        self.data['fraud_bool']=self.data['fraud_bool'].astype('category')
        self.data['income']= pd.Categorical(round(self.data['income'],2) ,categories=sorted(round(self.data['income'],2).unique()), ordered=True)
        self.data['customer_age']= pd.Categorical(self.data['customer_age'] ,categories=[10,20,30,40,50,60,70,80,90], ordered=True)
        self.data['email_is_free']=self.data['email_is_free'].astype('category')
        self.data['phone_home_valid']=self.data['phone_home_valid'].astype('category')
        self.data['phone_mobile_valid']=self.data['phone_mobile_valid'].astype('category')
        self.data['has_other_cards']=self.data['has_other_cards'].astype('category')
        self.data['foreign_request']=self.data['foreign_request'].astype('category')
        self.data['keep_alive_session']=self.data['keep_alive_session'].astype('category')
        self.data['source']=self.data['source'].astype('category')
        self.data['device_distinct_emails_8w']=self.data['device_distinct_emails_8w'].astype('category')
        
    
    def check_duplicates(self):
        if self.data.duplicated().sum()==0:
            return 'this data has no duplicates'
        else:
            number_of_duplicates=self.data.duplicated().sum()
            message='this data has '+str(number_of_duplicates)+' duplicates'
            return message
    
    def count_datatypes(self):

        dtype_dict={}
        dtype_dict['category']=len(self.data.select_dtypes('category').columns)
        dtype_dict['object']=len(self.data.select_dtypes('object').columns)
        dtype_dict['int']=len(self.data.select_dtypes('int').columns)
        dtype_dict['flaot']=len(self.data.select_dtypes('float').columns)

        return dtype_dict
    
    def get_data(self):
        return self.data
    
    def get_columns_with_possibe_missing_values(self):
        columns_with_possibe_missing=[]
        for column in self.data.columns:

            try:
                print(column,self.data[column].value_counts().loc[-1])
                columns_with_possibe_missing.append(column)
            except:
                pass
        return columns_with_possibe_missing
    
    def put_nulls(self):
        columns=self.get_columns_with_possibe_missing_values()
        for column in columns:
            self.data[column] = self.data[column].replace(-1, np.nan)
        return self.data.isnull().sum()
        
    def describe(self,flag):
        if flag=='numerical':
            return self.data.describe(percentiles=[0.01,0.25,0.75,0.99])
        if flag=='categorical':
            return self.data.describe(include=['object','category'])
        
    def plot_corr_heatmap(self):
        corr = self.data.corr()
        plt.figure(figsize=(15,15))
        sns.heatmap(round(corr,2), annot=True, cmap='coolwarm')
        sns.set(font_scale=1.2)
        plt.show()
        return None
        
    def identify_columns_to_drop(self,corr, threshold=0.8):

        columns_to_drop = []
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > threshold:
                    colname = corr.columns[i]
                    if colname != self.target_col and colname not in columns_to_drop:
                        columns_to_drop.append(colname)
        return columns_to_drop 
    
    
    def delete_columns(self, column_list):
        columns_deleted = []
        columns_not_found = []
        for column_name in column_list:
            if column_name in self.data.columns:
                self.data.drop(column_name, axis=1, inplace=True)
                columns_deleted.append(column_name)
            else:
                columns_not_found.append(column_name)
        if columns_deleted:
            message = f"Columns {', '.join(columns_deleted)} deleted successfully."
        else:
            message = "No columns deleted."
        if columns_not_found:
            message += f" Columns {', '.join(columns_not_found)} not found in the DataFrame."
        return message

    def plot_cramer(self):

        cramersv = am.CramersV(self.data) 
        cor_df=cramersv.fit()

        annot_labels = np.empty_like(cor_df, dtype=str)
        annot_mask = cor_df >= 0.8
        annot_labels[annot_mask] = 'T' 
        # Plot hearmap with the annotations
        fig, ax = plt.subplots(figsize=(30, 10))
        sns.heatmap(round(cor_df,2), annot=True, fmt='')
        return None
        
    def find_numerical_columns_with_possible_outliers(self,z_score_thresh=3, iqr_thresh=1.5):
        columns_which_may_contain_outliers=[]
        for column in self.data.select_dtypes(['int','float']).columns:
            
            is_normal = stats.normaltest(self.data[column])[1] > 0.05

            if is_normal:

                z_scores = np.abs(stats.zscore(self.data[column]))
                outliers = self.data[column][z_scores > z_score_thresh]
            else:

                q1, q3 = np.percentile(self.data[column], [25, 75])
                iqr = q3 - q1
                lower_fence = q1 - iqr_thresh * iqr
                upper_fence = q3 + iqr_thresh * iqr
                outliers = self.data[column][(self.data[column]< lower_fence) | (self.data[column] > upper_fence)]


            if len(outliers) > 0:
                columns_which_may_contain_outliers.append(column)
        return columns_which_may_contain_outliers

    def find_categorical_columns_with_possible_extremes(self, low_freq_thresh=0.05, high_freq_thresh=0.5):
        
        outlier_cols = []


        for column in self.data.select_dtypes(include=['object', 'category']):
            # Calculate the frequency of each category
            freq = self.data[column].value_counts(normalize=True)

            # Identify categories with low frequency
            low_freq_cats = freq[freq < low_freq_thresh].index.tolist()
            if len(low_freq_cats) > 0:
                outlier_cols.append(column)

            # Identify categories with high frequency
            high_freq_cats = freq[freq > high_freq_thresh].index.tolist()
            if len(high_freq_cats) > 0:
                outlier_cols.append(column)

        return list(set(outlier_cols))
    
    def get_skewed_columns(self, threshold=1):
        # Select the numerical columns
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64'])

        # Calculate the skewness for each column
        skewness_values = numerical_columns.skew()

        # Filter columns based on the threshold
        skewed_columns = skewness_values[skewness_values > threshold].index.tolist()

        return skewed_columns

    def draw_skewness_barchart(self):
        # Select the numerical columns
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64'])

        # Calculate the skewness for each column
        skewness_values = numerical_columns.skew()
        # Sort the skewness values
        skewness_values_sorted = skewness_values.sort_values(ascending=True)
        print(skewness_values_sorted)

        # Create a bar chart of the skewness values
        fig, ax = plt.subplots(figsize=(10, 8)) # Increase the figure size
        ax.bar(skewness_values_sorted.index, skewness_values_sorted.values)
        ax.tick_params(axis='x', rotation=90) # Rotate the xticks
        ax.set_title('Skewness Values')
        ax.set_xlabel('Column')
        ax.set_ylabel('Skewness')
        # Add the threshold line
        ax.axhline(y=1, color='r', linestyle='--')
        plt.show()

        return self.get_skewed_columns()

    def plot_numerical_columns(self):
        numerical_columns = self.data.select_dtypes(include=['float', 'int'])
        for column in numerical_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.data[column].dropna(), kde=True)
            plt.title(f'Numerical Column: {column}')
            plt.show()
            
    def plot_categorical_columns(self):
        categorical_columns = self.data.select_dtypes(include=['object', 'category'])
        for column in categorical_columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(data=self.data, x=column)
            plt.title(f'Categorical Column: {column}')
            plt.xticks(rotation=90)
            plt.show()
            
    def handle_null_values(self, method='mode', groupby_cols=None):
        if method not in ['mode', 'mean', 'median', 'groupby']:
            return "Invalid method. Available options: 'mode', 'mean', 'median', 'groupby'."

        if method == 'groupby' and groupby_cols is None:
            return "Groupby columns must be provided for the 'groupby' method."

        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                if method == 'mode':
                    fill_value = self.data[column].mode()[0]
                elif method == 'mean':
                    fill_value = self.data[column].mean()
                elif method == 'median':
                    fill_value = self.data[column].median()
                elif method == 'groupby':
                    fill_value = self.data.groupby(groupby_cols)[column].transform('mean')

                self.data[column].fillna(fill_value, inplace=True)

        return "Null values handled successfully."
    
    def apply_log_transform(self, column_name):
        if column_name in self.data.columns:
            self.data[column_name] = np.log1p(self.data[column_name])
            return f"Log transform applied to column '{column_name}' successfully."
        else:
            return f"Column '{column_name}' does not exist in the DataFrame."

        
         
    def handle_outliers(self,columns, method='replace', value=None):
        # Get the column of self.numerical_data to be processed
        for column in columns:
            column_data = self.data[column]

            # Perform normality test on the self.numerical_data
            _, p_value = shapiro(column_data)

            # If p-value is less than 0.05, self.numerical_data is not normal
            if p_value < 0.05:
                # Use IQR method to handle outliers
                q1 = column_data.quantile(0.25)
                q3 = column_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                if method == 'replace':
                    column_data = np.clip(column_data, lower_bound, upper_bound)
                    self.data[column] = column_data
                elif method == 'delete':
                    # Delete all records from self.numerical_data, not just the outliers
                    self.data = self.data[(self.data[column] > lower_bound) & (self.data[column] < upper_bound)]

            else:
                # Use Z-score method to handle outliers
                z_scores = np.abs((column_data - column_data.mean()) / column_data.std())
                outlier_mask = z_scores > 3
                non_outlier_values = column_data[~outlier_mask]
                median_non_outliers = non_outlier_values.mean()
                
                if method == 'replace':
                    column_data[outlier_mask] = np.sign(column_data[outlier_mask]) * median_non_outliers
                    self.data[column] = column_data
                elif method == 'delete':
                    # Delete all records from self.numerical_data, not just the outliers
                    self.data = self.data[~outlier_mask]


            # If method is 'replace' and a value is provided, replace outliers with the value
            if method == 'replace' and value is not None:
                column_data[outlier_mask] = value
            # Update the column in the original self.numerical_data with the processed column
                self.data[column] = column_data

        return self.data