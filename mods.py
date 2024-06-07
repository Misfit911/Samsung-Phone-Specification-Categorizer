#import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary modules for df for Modelling
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Imbalanced correction
from imblearn.over_sampling import SMOTE

# Models
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# Evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score

# Plot_roc_curves
from IPython.display import display, HTML
from sklearn.metrics import roc_curve, auc

# Filter warnings
import warnings
warnings.filterwarnings('ignore')
# Class that contains functions used for data loading and previewing features
class DataSourcing:
    def __init__(self):
        pass
    
    def open_file(self, path):
        df = pd.read_csv(path)
        df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)  # Rename the column
        return df
    def dataframe_details(self,df):
        """
        Print details of the dataframe.
        Parameters:
        df (DataFrame): The dataframe to be analyzed.
        Returns:
        None
        """
        print("=" * 30)
        print(f"DATAFRAME SHAPE: {df.shape}")
        print("=" * 30, "\n\n")
        print("=" * 15)
        print(f"DATAFRAME HEAD:")
        print("=" * 15)
        print(f"{df.head()}")
        print("=" * 70, "\n\n")
        print("=" * 30)
        print(f"DATAFRAME COLUMNS INFO:")
        print("=" * 30)
        print(f"{df.info()}")
        print("=" * 70, "\n\n")
        print("=" * 30)
        print(f"DATAFRAME KEY STATISTICS:")
        print("=" * 30)
        print(f"{df.describe().transpose()}")
        print("=" * 70, "\n\n")

# Class that contains functions used for data preparation
class DataPreprocessing(DataSourcing):
    def __init__(self):
        super().__init__()
        """Data Preprocessing class that inherits from the data sourcing class.
        Contains functions to be used to check certain aspects in the data for cleaning.
        Checks for duplicates, nulls and outliers
        """

    def check_duplicates(self, data):
        duplicates = data[data.duplicated()].shape[0]
        print("There are {} duplicates in the data.".format(duplicates))
        
    def check_null_values(self, data):
        null_values = data.isnull().sum()
        print(null_values)
        print("=" * 40)
        print("List of columns with missing values:")
        print("=" * 40)
        return null_values[null_values > 0].index.tolist()

    # Drop columns no longer applicable
    def drop_columns(self, data, columns):
        for col in columns:
            data.drop(columns=col, inplace=True)

    def create_version_category(self, data):
        data['Android_version_cleaned'] = data['Android_version'].str.replace(r'\s*\(.*\)', '', regex=True)
        
        def convert_version(version):
            version_str = str(version)
            parts = version_str.split('.')
            return float(parts[0]) + float(parts[1]) / 10 if len(parts) > 1 else float(parts[0])
        
        data['Android_version_numeric'] = data['Android_version_cleaned'].apply(convert_version)
        data['Version_category'] = np.where(data['Android_version_numeric'] >= 11, 'Android 11+', 'Older Versions')
        
    def create_fast_charging_column(self, data):
            data['fast_charging'] = data['fast_charging'].apply(lambda x: 'supports fast charging' if pd.notna(x) else 'does not support fast charging')

    # Method for imputing missing values
    def impute_missing_values(self, data):
        missing_values = data.isna().sum()
        missing_value_columns = missing_values[missing_values > 0].index.tolist()

        missing_numeric_cols = data[missing_value_columns].select_dtypes(include=['number']).columns
        missing_object_cols = data[missing_value_columns].select_dtypes(include=['object']).columns

        numeric_imputer = SimpleImputer(strategy='mean')
        object_imputer = SimpleImputer(strategy='most_frequent')

        data[missing_numeric_cols] = numeric_imputer.fit_transform(data[missing_numeric_cols])
        data[missing_object_cols] = object_imputer.fit_transform(data[missing_object_cols])

        return data.isna().sum()

    def create_rating_category(self, data):
        data['Rating_binary'] = (data['Rating'] >= 4).astype(int)
        data['rating_category'] = data['Rating_binary'].map({1: 'Good', 0: 'Not Good'})

    def create_spec_score_category(self, data):
        data['Spec_score_binary'] = (data['Spec_score'] >= 70).astype(int)
        data['Spec_score_category'] = data['Spec_score_binary'].map({1: 'High-Spec', 0: 'Low-Spec'})

    def create_price_category(self, data):
        data['Price'] = data['Price'].str.replace(',', '').astype(int)
        upper_quartile_price = data['Price'].quantile(0.75)
        data['Price_binary'] = (data['Price'] <= upper_quartile_price).astype(int)
        data['Price_category'] = data['Price_binary'].map({1: 'Expensive', 0: 'Affordable'})

    def create_company_category(self, data):
        data['Company_category'] = np.where(data['company'] == 'Samsung', 'Samsung', 'Other Brands')

# Class for Data Visualization and Analysis
class DataAnalysis(DataPreprocessing,DataSourcing):
    
    def __init__(self,data=None):
        self.data = data

    def plot_boxplots(self, data, numeric_columns):
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(numeric_columns, 1):
            plt.subplot(4, 3, i)
            sns.boxplot(x=data[column])
            plt.title(f'{column.capitalize()}')
            plt.xlabel(column)
        
        plt.tight_layout()
        plt.show()

    def plot_histograms(self, data, numeric_columns):
        num_columns = len(numeric_columns)
        layout_rows = (num_columns // 3) + (num_columns % 3 > 0)
        data[numeric_columns].hist(figsize=(15, 2.5 * layout_rows), layout=(layout_rows, 3))
        plt.suptitle('Distribution of Numerical Features', y=1.02)
        plt.show()

    def plot_countplots(self, data, relevant_cols):
        num_rows = 3
        num_columns = 2
        plt.figure(figsize=(15, 10))

        for i, column in enumerate(relevant_cols, 1):
            plt.subplot(num_rows, num_columns, i)
            sns.countplot(x=data[column])
            plt.title(f'{column.capitalize()}')
            plt.xlabel(column)
            plt.ylabel('Count')

        plt.tight_layout()
        plt.show()

    # Check the distribution of the column variables
    def check_normal_distribution(self, data, exclude_columns=[]):
        for col in data.columns:
            if col in exclude_columns:
                continue  # Skip the column if it is in the exclude_columns list
            
            if data[col].dtype == 'object' or data[col].nunique() < 20:
                # If the column is categorical or has less than 20 unique values, use countplot
                custom_palette = ["blue", "orange", "green", "red", "purple", "brown"]
                sns.countplot(data=data, x=col, palette=custom_palette)
                plt.title(f"Distribution of '{col}'")
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=90)
                plt.show()
            elif pd.api.types.is_numeric_dtype(data[col]):
                sns.histplot(data[col], kde=True, stat="density")
                plt.title(f"Distribution of '{col}'")
                plt.xlabel(col)
                plt.ylabel('Density')
                plt.show()

    def encode_categorical_features(self, data):
        categorical_features = ['Battery', 'No_of_sim', 'Ram', 'company', 'Company_category', 'Inbuilt_memory', 'Processor']
        data = pd.get_dummies(data, columns=categorical_features)
        return data

    def correlation(self, data, column=None, rank=False):
        '''
        Performs correlation matrix, drops non-numeric object types.
        
        Parameters:
        data (pd.DataFrame): The dataframe to perform correlation on.
        column (str, optional): The column name to find correlation against. Default is None.
        rank (bool, optional): Whether to rank the correlations. Default is False.
        
        Returns:
        pd.Series or pd.DataFrame: Correlation series or ranked correlation series.
        '''
        # Redundant check. Makes sure all columns of object type are dropped
        data_object = data.select_dtypes(include='object')
        print(f'The following columns were dropped due to being object types:\n{data_object.columns.tolist()}')
        data = data.drop(data_object.columns, axis=1)
        
        if rank:
            if column:
                try:
                    data_corr = data.corr()[column]
                    data_corr = data_corr.drop(column)
                    print(f'Ranked correlation to {column}')
                    return data_corr.rank(ascending=False).sort_values()
                except KeyError:
                    print(f'Column "{column}" not found in dataframe.')
                    return None
            else:
                data_corr = data.corr()
                return data_corr.rank(ascending=False)
        else:
            if column:
                try:
                    data_corr = data.corr()[column]
                    data_corr = data_corr.drop(column)
                    print(f'Correlation to {column}')
                    return data_corr.sort_values(ascending=False)
                except KeyError:
                    print(f'Column "{column}" not found in dataframe.')
                    return None
            else:
                data_corr = data.corr()
                return data_corr

# Class that contains functions used for data preparation
class DataModeling(DataAnalysis,DataPreprocessing,DataSourcing):
    lr = LogisticRegression()
    dc = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier(n_neighbors=5)

    def __init__(self,data=None):
        self.data = data

    def split_data(self, data, target, test_size=0.3, random_state=42):
        X = data.drop([target], axis=1)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def apply_smote(self, X_train, y_train):
        smote = SMOTE(sampling_strategy=0.75, k_neighbors=5, random_state=None)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled

    def modelplotting_evaluation(self, X_train, X_test, y_train, y_test):
        Results = {'Model': [], 'Accuracy': [], 'Recall': [], 'Precision': [], 'F1': []}

        lr = LogisticRegression(random_state=42)
        dc = DecisionTreeClassifier(random_state=42)
        rf = RandomForestClassifier(random_state=42)
        knn = KNeighborsClassifier(n_neighbors=5)

        # fitting and prediction
        model_list = [lr, dc, rf, knn]

        for model in model_list:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print('Model:', model)
            print('-' * 80)
            print('Confusion matrix:\n\n', confusion_matrix(y_test, y_pred))
            print('\nClassification report:', classification_report(y_test, y_pred))
            print('Accuracy:', accuracy_score(y_test, y_pred))
            print('Precision:', precision_score(y_test, y_pred))
            print('Recall:', recall_score(y_test, y_pred))
            print('F1 score:', f1_score(y_test, y_pred))
            print('-' * 80)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            print()
            print("Cross-Validation Scores:", cv_scores)
            print("Mean CV Accuracy:", np.mean(cv_scores))
            print()
            print('-' * 80)

            R = {'Model': str(model),
                'Accuracy': accuracy_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred)
                }

            Results['Model'].append(R['Model'])
            Results['Accuracy'].append(R['Accuracy'])
            Results['Recall'].append(R['Recall'])
            Results['Precision'].append(R['Precision'])
            Results['F1'].append(R['F1'])

        return pd.DataFrame(Results)

    def plot_roc_curves(self, models, X_train_resampled, y_train_resampled, X_test, y_test, X_train_params=None, y_train_params=None):
        '''
        Plots ROC curves for the given models.

        Parameters:
        models (list): List of models to plot ROC curves for.
        X_train_resampled (array-like): Resampled training features.
        y_train_resampled (array-like): Resampled training labels.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        X_train_params (array-like, optional): Custom parameters for X_train_resampled.
        y_train_params (array-like, optional): Custom parameters for y_train_resampled.
        '''
        # Use custom parameters if provided, otherwise use default
        X_train = X_train_params if X_train_params is not None else X_train_resampled
        y_train = y_train_params if y_train_params is not None else y_train_resampled

        plt.figure(figsize=(10, 8))
        
        for model in models:
            model.fit(X_train, y_train)
            
            # Predict probabilities for positive class
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Compute ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{model.__class__.__name__} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.show()
