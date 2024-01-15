import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import tkinter as tk
pio.templates.default = "plotly_white"
data = pd.read_csv('train1.csv')
print(data.head(2))


# Which features affects credit score most checking
# fig = px.box(data,
#              x="Credit_Score",
#              y="Monthly_Balance",
#              color="Credit_Score",
#              title="Credit Scores Based on Monthly Balance Left",
#              color_discrete_map={'Poor': 'red',
#                                  'Standard': 'yellow',
#                                  'Good': 'green'})
# fig.update_traces(quartilemethod="exclusive")
# fig.show()

# Converting Credit Mix model to numeric for machine learning model
data["Credit_Mix"] = data["Credit_Mix"].map({
    "Standard": 1,
    "Good": 2,
    "Bad": 0,
})

print(data["Credit_Mix"].value_counts())

# Creation of Machine Learning Model
x = np.array(data[["Monthly_Inhand_Salary",
                   "Num_Bank_Accounts", "Num_Credit_Card",
                   "Num_of_Loan",
                   "Delay_from_due_date", "Num_of_Delayed_Payment",
                   "Credit_Mix", "Outstanding_Debt",
                   "Credit_History_Age", "Monthly_Balance"]])
y = np.array(data["Credit_Score"])

# Splitting the value of X & Y to different text variables
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.33,
                                                random_state=42)
# Random Forest Algorithm for prediction of Credit Score
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# Checking the Model based on Given Input Value by user
print("Credit Score Prediction : ")
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))


features = np.array([[b, c, d, f, g, h, i, j, k, l]])
score = model.predict(features)
print("Your credit Status: " + score[0])
# Checking the Status Based on Status Value
if score[0] == 'Good':
    credit_score = 651
elif score[0] == 'Standard':
    credit_score = 649
else:
    credit_score = 350

# Define the score ranges and labels
score_ranges = [0, 500, 650, 800]
labels = ["Poor(0-500)", "Standard(501-700)", "Good(701-850+)"]

# Determine the index of the score range that the credit score falls into
index = sum(score <= credit_score for score in score_ranges)

# Define the sizes of each pie slice
sizes = [1 / 3] * 3
sizes[index - 1] = 0.4

# Define the explode parameter to explode the selected slice
explode = [0.1 if i == index - 1 else 0 for i in range(3)]

# Create the pie chart
plt.pie(sizes, explode=explode, labels=labels, startangle=90)

# Add a title to the chart
plt.title("Credit Score Pie Chart")

# Show the chart
plt.show()

