from flask import Flask, request, jsonify
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report

app = Flask(__name__)

# Load the pre-trained machine learning model
with open('new_rf_model.pickle', 'rb') as file:
    new_rf_model = pickle.load(file)

# Load the feature data
with open('features_data.json', 'r') as file:
    features_data = json.load(file)

class_labels = ['crop1', 'crop2', 'crop3']  # Replace with your actual class labels

@app.route('/api/send_data', methods=['POST'])
def receive_data():
    data = request.json
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    rainfall = data.get('region')

    print("Received data - Temperature:", temperature, "Humidity:", humidity, "Region:", rainfall)

    # Call the last_part function with the received data
    crop_recommendation = last_part(temperature, humidity, rainfall)

    return jsonify({'recommended_crop': crop_recommendation})







from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier


df = pd.read_csv('C:/Users/User/Desktop/machine learning/precision agricluture/Crop_recommendation.csv')
df


def initial_part():
    df.info()


df.columns

df['label'].unique()

df['label'].value_counts()

# df['label'].value_counts().plot(kind='pie',autopct="%.2f%%",hatch=['**O', 'oO', 'O.O', '.||.'],shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9}, startangle=90)
# plt.show()


# sns.histplot(df['N'])
# plt.title('Histogram of Nitrogen')
# plt.show()

# sns.histplot(df['P'])
# plt.title('Histogram of Phosphorus')
# plt.show()


# sns.histplot(df['K'])
# plt.title('Histogram of potassium')
# plt.show()

# sns.histplot(df['temperature'],color='Purple')
# plt.title('Histogram of Temperature')
# plt.show()


# sns.histplot(df['humidity'],color='Yellow')
# plt.title('Histogram of Humidity')
# plt.show()


# sns.histplot(df['ph'],color='Red')
# plt.title('Histogram of PH')
# plt.show()


# sns.histplot(df['rainfall'],color='Cyan')
# plt.title('Histogram of Rainfall')
# plt.show()


plt.figure(figsize=(12,12))
i=1
for col in df.iloc[:,:-1]:
    plt.subplot(3,3,i)
    sns.kdeplot(df[col])
    i+=1


    import scipy.stats as sm


    plt.figure(figsize=(12,12))
i=1
for col in df.iloc[:,:-1]:
    plt.subplot(3,3,i)
    sm.probplot(df[col],dist='norm',plot=plt)
    i+=1

    plt.figure(figsize=(12,12))
i=1
for col in df.iloc[:,:-1]:
    plt.subplot(3,3,i)
    df[[col]].boxplot()
    i+=1

    df.iloc[:,:-1].skew()

    class_labels = df['label'].unique().tolist()
class_labels

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

df['label']


class_labels = le.classes_
class_labels


df


x = df.drop('label',axis=1)
y = df['label']


    


def part_2():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=True)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)

    y_pred = rf_model.predict(x_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print()
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Hyperparameter tuning using RandomizedSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_model = RandomForestClassifier()
    random_search = RandomizedSearchCV(rf_model, param_distributions=param_grid, n_iter=10, cv=3, random_state=42)
    random_search.fit(x_train, y_train)

    # Get the best model from the search
    best_rf_model = random_search.best_estimator_

    # Model evaluation on Test Data

    # Update this line to use best_rf_model instead of rf_model.best_estimator_
    new_rf_model = best_rf_model

    y_pred = new_rf_model.predict(x_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print()
    print("Classification Report:\n", classification_report(y_test, y_pred))

    y_pred_train = new_rf_model.predict(x_train)

    print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
    print()
    print("Classification Report:\n", classification_report(y_train, y_pred_train))

    # ... (the rest of your code)
    # ...

    # Saving the model and feature data


    # ...

    # Make sure to define features_data before this point in your code
    features_data = {'columns': list(x.columns)}

    with open('new_rf_model.pickle', 'wb') as file:
        pickle.dump(new_rf_model, file)

    with open('features_data.json', 'w') as file:
        json.dump(features_data, file)

    len(features_data['columns'])







def last_part(temperature, humidity, rainfall):
    # Taking User Inputs
    test_series = pd.Series(np.zeros(len(features_data['columns'])), index=features_data['columns'])

    # Assuming 'temperature', 'humidity', and 'region' are features in your dataset
    test_series['N'] = 90
    test_series['P'] = 42
    test_series['K'] = 43
    test_series['temperature'] = temperature
    test_series['humidity'] = humidity
    test_series['ph'] = 6.5
    test_series['rainfall'] = rainfall

    test_series

    output = new_rf_model.predict([test_series])[0]
    recommended_crop = class_labels[output]

    print("Recommended Crop:", recommended_crop)

    return recommended_crop

if __name__ == '__main__':
    app.run(debug=True)






