import numpy as np
import pandas as pd

# Setting display options for pandas
pd.set_option("display.max.columns", 100)

# Importing necessary libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Suppressing warnings
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
data = pd.read_csv(DATA_URL + "adult.data.csv")

# Display the first few rows (optional, for verification)
print(data.head())

# ----- QUESTION 1 -----
# How many men and women (sex feature) are represented in this dataset?
print("\nQuestion 1:")
print(data['sex'].value_counts())

# ----- QUESTION 2 -----
# What is the average age (age feature) of women?
print("\nQuestion 2:")
average_age_women = data[data['sex'] == 'Female']['age'].mean()
print(f"Average age of women: {average_age_women}")

# ----- QUESTION 3 -----
# What is the percentage of German citizens (native-country feature)?
print("\nQuestion 3:")
germans = data[data['native-country'] == 'Germany'].shape[0]
total_people = data.shape[0]
percentage_germans = (germans / total_people) * 100
print(f"Percentage of German citizens: {percentage_germans:.2f}%")

# ----- QUESTION 4-5 -----
# What are the mean and standard deviation of age for those who earn more than 50K per year and those who earn less?
print("\nQuestion 4-5:")
mean_age_above_50k = data[data['salary'] == '>50K']['age'].mean()
std_age_above_50k = data[data['salary'] == '>50K']['age'].std()
mean_age_below_50k = data[data['salary'] == '<=50K']['age'].mean()
std_age_below_50k = data[data['salary'] == '<=50K']['age'].std()
print(f"Mean age (above 50K): {mean_age_above_50k:.2f}, Std (above 50K): {std_age_above_50k:.2f}")
print(f"Mean age (below 50K): {mean_age_below_50k:.2f}, Std (below 50K): {std_age_below_50k:.2f}")

# ----- QUESTION 6 -----
# Is it true that people who earn more than 50K have at least high school education?
print("\nQuestion 6:")
high_education = ['Bachelors', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Masters', 'Doctorate']
higher_earners = data[data['salary'] == '>50K']
education_check = all(higher_earners['education'].isin(high_education))
print(f"All higher earners have at least high school education: {education_check}")

# ----- QUESTION 7 -----
# Display age statistics for each race (race feature) and each gender (sex feature). Find the maximum age of men of Amer-Indian-Eskimo race.
print("\nQuestion 7:")
age_stats = data.groupby(['race', 'sex'])['age'].describe()
print(age_stats)
max_age_amer_indian_eskimo = data[(data['race'] == 'Amer-Indian-Eskimo') & (data['sex'] == 'Male')]['age'].max()
print(f"Max age of Amer-Indian-Eskimo men: {max_age_amer_indian_eskimo}")

# ----- QUESTION 8 -----
# Among whom is the proportion of those who earn a lot (>50K) greater: married or single men?
print("\nQuestion 8:")
married_statuses = ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
data['marital_category'] = np.where(data['marital-status'].isin(married_statuses), 'Married', 'Single')
men_data = data[data['sex'] == 'Male']
married_proportion = men_data[men_data['marital_category'] == 'Married']['salary'].value_counts(normalize=True)['>50K']
single_proportion = men_data[men_data['marital_category'] == 'Single']['salary'].value_counts(normalize=True)['>50K']
print(f"Proportion of >50K earners among married men: {married_proportion:.2f}")
print(f"Proportion of >50K earners among single men: {single_proportion:.2f}")

# ----- QUESTION 9 -----
# What is the maximum number of hours a person works per week? How many people work that many hours, and what is the percentage of those who earn a lot (>50K) among them?
print("\nQuestion 9:")
max_hours = data['hours-per-week'].max()
num_people_max_hours = data[data['hours-per-week'] == max_hours].shape[0]
percent_earning_above_50k = (data[(data['hours-per-week'] == max_hours) & (data['salary'] == '>50K')].shape[0] / num_people_max_hours) * 100
print(f"Max hours worked per week: {max_hours}")
print(f"Number of people working {max_hours} hours per week: {num_people_max_hours}")
print(f"Percentage of those earning >50K among them: {percent_earning_above_50k:.2f}%")

# ----- QUESTION 10 -----
# Count the average time of work (hours-per-week) for those who earn a little and a lot (salary) for each country. What will these be for Japan?
print("\nQuestion 10:")
avg_hours_per_country = data.groupby(['native-country', 'salary'])['hours-per-week'].mean()
print(avg_hours_per_country.loc['Japan'])
