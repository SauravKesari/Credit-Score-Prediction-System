import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('train1.csv')

# Converting Credit Mix model to numeric for machine learning model
data["Credit_Mix"] = data["Credit_Mix"].map({"Standard": 1, "Good": 2, "Bad": 0})

# Creation of Machine Learning Model
x = np.array(data[["Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan",
                   "Delay_from_due_date", "Num_of_Delayed_Payment", "Credit_Mix", "Outstanding_Debt",
                   "Credit_History_Age", "Monthly_Balance"]])
y = np.array(data["Credit_Score"])

# Splitting the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)

# Random Forest Algorithm for prediction of Credit Score
model = RandomForestClassifier()
model.fit(xtrain, ytrain)


def predict_credit_score():
    # Get the user input values from the entry fields
    try:
        b = float(monthly_salary_entry.get())
        c = float(num_bank_accounts_entry.get())
        d = float(num_credit_cards_entry.get())
        f = float(num_loans_entry.get())
        g = float(days_delayed_entry.get())
        h = float(num_delayed_payments_entry.get())
        i = credit_mix_entry.get()
        j = float(outstanding_debt_entry.get())
        k = float(credit_history_age_entry.get())
        l = float(monthly_balance_entry.get())

        features = np.array([[b, c, d, f, g, h, i, j, k, l]])
        score = model.predict(features)
        credit_status = score[0]

        # Checking the Status Based on Status Value
        if credit_status == 'Good':
            credit_score = 651
        elif credit_status == 'Standard':
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
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, explode=explode, labels=labels, startangle=90, colors=['red', 'yellow', 'green'])
        plt.title("Credit Score Pie Chart")
        plt.show()
    except ValueError:
        messagebox.showerror("Error", "Invalid input values. Please enter valid numeric values.")


# Create the main window
window = tk.Tk()
window.geometry("500x500")  # Set the size of the window (width x height)
window.configure(bg="#C2DEDC")  # Set the background color of the window

window.title("Credit Score Prediction")

# Create labels and entry fields
labels = ["Monthly Inhand Salary:", "Number of Bank Accounts:", "Number of Credit Cards:",
          "Number of Loans:", "Average number of days delayed by the person:", "Number of delayed payments:",
          "Credit Mix (Bad: 0, Standard: 1, Good: 2):", "Outstanding Debt:", "Credit History Age:",
          "Monthly Balance:"]

entries = []

for idx, label_text in enumerate(labels):
    label = tk.Label(window, text=label_text, width=30, anchor='w', fg='black')
    label.grid(row=idx, column=0, padx=10, pady=10, sticky='w')

    entry = tk.Entry(window, width=10, bd=2)
    entry.grid(row=idx, column=1, padx=10, pady=10, ipadx=5)

    entries.append(entry)

# Assign entry fields to individual variables
monthly_salary_entry = entries[0]
num_bank_accounts_entry = entries[1]
num_credit_cards_entry = entries[2]
num_loans_entry = entries[3]
days_delayed_entry = entries[4]
num_delayed_payments_entry = entries[5]
credit_mix_entry = entries[6]
outstanding_debt_entry = entries[7]
credit_history_age_entry = entries[8]
monthly_balance_entry = entries[9]

# Create the predict button
predict_button = tk.Button(window, text="Predict", command=predict_credit_score, fg='black')
predict_button.grid(row=len(labels), columnspan=2, padx=10, pady=10)

# Run the main window loop
window.mainloop()
