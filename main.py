import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re

# Project
# Dataset - https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_19_india.csv
# Repository Link - https://github.com/yogamoorthyr/UCDPA_YOGAMOORTHYR/blob/master/main.py

# 1. Real-world scenario
# ○ The project should use a real-world dataset and include a reference of their source in the report (10)
# Dataset - https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_19_india.csv


# 2. Importing data
# ○ Your project should make use of one or more of the following: Relational database, API or web scraping (10)
# API or Relational database (Unable to get the sql data nor test it due to access challenge, hence, provided only with codes)
# from sqlalchemy import create_engine
# engine = create_engine ('sqlite:///-----.sqlite')
# df = pd.read_sql_query("SELECT * FROM ----- WHERE ----- > 10 ORDER By -----", engine)
# Using INNER JOIN function
# df = pd.read_sql_query ("SELECT -----, ----- FROM ----- INNER JOIN ----- on -----.----- = -----.-----", engine)
# print (df.head)

# For illustration purpose only (due to data unavailability)
# Read csv data into pandas dataframes
# covid_data = ut.read_file(src_type='csv', name=covid_data.csv')
# covid_vaccine = ut.read_file(src_type='csv', name=covid_vaccine.csv')

# Read the same files into pandas dataframe from MySQL database
# covid_data = ut.read_file(src_type='db', name='covid_data')
# covid_vaccine = ut.read_file(src_type='db', name='covid_vaccine')



# Web Scraping
# import requests
# NOTE: The below command works as expected but, has been set to non-execute position as it creates too cluster in results.  Please remove '#' to execute
# request = requests.get('https://www.kaggle.com/sudalairajkumar/covid19-in-india?select=covid_19_india.csv')
# print (request.text)
# print (type(request.text))
# print (type(request))

# ○ Import a CSV file into a Pandas DataFrame (10)
import pandas as pd
covid_data = pd.read_csv(r'C:\Users\Dell\PycharmProjects\pythonProject\UCDPA_YOGA MOORTHY R\covid_19_india.csv', index_col=0)
covid_vaccine = pd.read_csv(r'C:\Users\Dell\PycharmProjects\pythonProject\UCDPA_YOGA MOORTHY R\covid_vaccine_statewise.csv', index_col=0)
statewise_testing = pd.read_csv(r'C:\Users\Dell\PycharmProjects\pythonProject\UCDPA_YOGA MOORTHY R\StatewiseTestingDetails.csv', index_col=0)

print (covid_data)
print (covid_vaccine)
print (statewise_testing)

# Using "iloc" function
print (covid_data.iloc[0:5])

# Using "loc" function
print (covid_data.loc[1])

print (type(covid_data))
print (type(covid_vaccine))
print (type(statewise_testing))

# Using Header function
# print (covid_data.head(5))
# print (covid_vaccine.head(5))
# print (statewise_testing.head(5))

# Info on every csv file
print (covid_data.info())
print (covid_vaccine.info())
print (statewise_testing.info())

# Using Describe function
print (covid_data.describe())
print (covid_vaccine.describe())
print (statewise_testing.describe())


# 3. Analysing data
# ○ Your project should use Regex to extract a pattern in data (10)
import re
p = re.compile(r'\W+')
p2 = re.compile(r'(\W+)')
p.split('This... is a test.')

p2.split('This... is a test.')



# ○ Replace missing values or drop duplicates (10)
missing_values_count = covid_data.isnull().sum()
print (missing_values_count)
missing_values_count = covid_vaccine.isnull().sum()
print (missing_values_count)
missing_values_count = statewise_testing.isnull().sum()
print (missing_values_count)

# Blanks & Hyphen thoroughly checked & cleansed in all 3 excel sheets (rought work data in "Run Results" word document)

import pandas as pd
import numpy as np
covid_data = covid_data.replace(to_replace='-', value = 0)
covid_data.fillna(0, inplace=True)
# To check whether the data was updated or not
print (covid_data['Cured'])

covid_vaccine = covid_vaccine.replace(to_replace='-', value = 0)
#covid_vaccine.fillna(0, inplace=True)
covid_vaccine.replace(np.nan, 0, inplace=True)
#covid_vaccine = covid_vaccine.replace(to_replace='NaN', value = 0)
# To check whether the data was updated or not
print (covid_vaccine['Sputnik V (Doses Administered)'])

statewise_testing = statewise_testing.replace(to_replace='-', value = 0)
#statewise_testing.fillna(0, inplace=True)
statewise_testing.replace(np.nan, 0, inplace=True)
#statewise_testing = statewise_testing.replace(to_replace='NaN', value = 0)
print (statewise_testing)

# ○ Make use of iterators (10)
# ○ Merge DataFrames (10)
covid_data.rename(columns={'State/UnionTerritory':'State'}, inplace=True)
# print (covid_data["State"])
cd_st_data = covid_data.merge(statewise_testing, on = 'State')
print (cd_st_data.head())


# 4. Python
# ○ Define a custom function to create reusable code (10)
def check_nulls(df):
    print('Columns containing NULLS are:-')
    return df.columns[df.isna().any()].tolist()
check_nulls(covid_data)

# ○ NumPy (10)
import numpy as np
np.array(covid_vaccine)
np_covid_vaccine = np.array(covid_vaccine)
print (np_covid_vaccine)
print (type(np_covid_vaccine))
# To view 1 row of data
print (np_covid_vaccine[0])
# Total rows & Columns in 2D using Numpy
print (np_covid_vaccine.shape)

#print (np_covid_vaccine[0:, 5])

# ○ Dictionary or Lists (10)
# 5. Machine Learning (60)
# ○ Predict a target variable with Supervised or Unsupervised algorithm
# ○ You are free to choose any algorithm
# ○ Perform hyper parameter tuning or boosting, whichever is relevant to your model. If it is not relevant, justify that in your report and Python comments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
#X = cd_st_data.drop(columns=['Sno','Time'], axis=1)
X = cd_st_data['Confirmed']
#X.drop(columns=['Sno', 'Time'], inplace=True)
y = cd_st_data['Positive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
model = XGBRegressor()
print("Training the XGBRegressor model on the train dataset")
model.fit(X_train, y_train)
# Note the training exercise was set to 20% on the data provided

# 13. predicting with the test data
predict_positivecases = model.predict(X_test)
print('print the target predicted using test data')
print(predict_positivecases)

predict_positivecases1 = model.predict(X_test)
print('print the target predicted using test data')
print(predict_positivecases1)



# 6. Visualise
# ○ Present two charts with Seaborn or Matplotlib (20)
import matplotlib.pyplot as plt
#plot.plt()
#plt.show()


import seaborn as sns
sns.countplot(cd_st_data['State'])
plt.show()
plt.savefig('State.png')
print('State Chart')

sns.displot(cd_st_data['Confirmed'])
plt.show()

# 7. Generate valuable insights
# ○ 5 insights from the project (20)
# More or less all the States has the same visibility


Insights
#(Point out at least 5 insights in bullet points)
# All the States has more or less the same with affected individuals
# Maharashtra, Karnataka, Tamilnadu, Delhi, Uttar Pradesh stands in first 5 positions in No. of Deaths
# Max number of people vaccinated states stands as Maharashtra, Uttar Pradesh, Gujarat, Rajasthan & West Bengal
# Suggestion to both Karnataka & Tamilnadu states for more vaccination
# The promising figures on Cured patients on top 5 states stands as Maharashtra, Karnataka, Kerala, Tamilnadu, Andhra Pradesh which seems to quite extra-ordinary for Andhra Pradesh & Tamilnadu even though their position remains as 9th & 10th towards vaccinated states.
