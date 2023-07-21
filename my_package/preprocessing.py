from my_package.data_cleaning import *
import matplotlib.pyplot as plt
import association_metrics as am
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE,SMOTENC,ADASYN
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import shapiro
class preprocessing:
    
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
        self.numerical_data=self.data.select_dtypes(['int','float'])
        self.categorical_data=self.data.select_dtypes(['object','category'])
        
    
    def set_data(self,data):
        self.data = data
        self.target_col = target_col
        self.numerical_data=self.data.select_dtypes(['int','float'])
        self.categorical_data=self.data.select_dtypes(['object','category'])
        
    def get_data(self):
        return self.concat_data(),self.target_col
    
    def scale_data(self, scaling_type, column):
        if scaling_type == 'standard':
            scaler = StandardScaler()
            self.numerical_data[column] = scaler.fit_transform(self.numerical_data[[column]])
        elif scaling_type == 'minmax':
            scaler = MinMaxScaler()
            self.numerical_data[column] = scaler.fit_transform(self.numerical_data[[column]])
        elif scaling_type == 'robust':
            scaler = RobustScaler()
            self.numerical_data[column] = scaler.fit_transform(self.numerical_data[[column]])
        else:
            print("Invalid scaling type")
            return None

        return self.numerical_data, scaler

    
    def encode_categorical_features(self, column, encoding_type='one-hot'):
        encoded_data = self.categorical_data[column].copy()
        encoder = None
        if encoding_type == 'label':
            le = LabelEncoder()
            encoded_data = le.fit_transform(encoded_data)
            encoder = le
            self.categorical_data[column] = encoded_data
        elif encoding_type == 'one-hot':
            ohe = OneHotEncoder()
            encoded_col = pd.DataFrame(ohe.fit_transform(self.categorical_data[column].values.reshape(-1, 1)).toarray(),index=self.categorical_data.index)
            encoded_col.columns = [column + '_' + str(val) for val in ohe.categories_[0]]
            self.categorical_data = pd.concat([self.categorical_data, encoded_col], axis=1)
            self.categorical_data.drop(columns=column, inplace=True,axis=1)  
            encoder = ohe
        else:
            print("Invalid encoding type")
            return None

        return self.categorical_data, encoder


    
    def concat_data(self):
        # Concatenate the numerical and categorical data
        self.data = pd.concat([pd.DataFrame(self.numerical_data), self.categorical_data], axis=1)

        # Return the concatenated dataframe
        return self.data
    
    def oversample_data(self):
        # Oversample the minority class using RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_oversampled, y_oversampled = ros.fit_resample(self.data, self.target_col)

        # Return the oversampled data
        return X_oversampled, y_oversampled
        
    def undersample_data(self):
        # Undersample the majority class using RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_undersampled, y_undersampled = rus.fit_resample(self.data, self.target_col)

        # Return the undersampled data
        return X_undersampled, y_undersampled
    
    def smote_data(self):
        # Oversample the minority class using SMOTE
        smote = SMOTE(random_state=42)
        X_oversampled, y_oversampled = smote.fit_resample(self.data, self.target_col)

        # Return the oversampled data
        return X_oversampled, y_oversampled
    
    def smotenc_data(self, categorical_features):
        # Oversample the minority class using SMOTENC
        smotenc = SMOTENC(categorical_features=categorical_features, random_state=42)
        X_oversampled, y_oversampled = smotenc.fit_resample(self.data, self.target_col)

        # Return the oversampled data
        return X_oversampled, y_oversampled
