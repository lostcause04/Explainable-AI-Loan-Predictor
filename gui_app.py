import os
print(os.listdir("Datasets_AI_project"))

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("Datasets_AI_project/train_u6lujuX_CVtuZ9i.csv")
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

# Features & target
X = data.drop("Loan_Status_Y", axis=1)
y = data["Loan_Status_Y"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# GUI window
root = tk.Tk()
root.title("Loan Approval Predictor (Explainable AI)")
root.geometry("500x500")

# Input fields dictionary
entries = {}

# Create input fields
for i, col in enumerate(X.columns[:6]):  # limiting for simplicity
    tk.Label(root, text=col).pack()
    entry = tk.Entry(root)
    entry.pack()
    entries[col] = entry

# Prediction function
def predict():
    try:
        user_input = []

        for col in X.columns[:6]:
            val = float(entries[col].get())
            user_input.append(val)

        # Fill rest with 0
        remaining = [0]*(len(X.columns)-6)
        final_input = np.array(user_input + remaining).reshape(1, -1)

        prediction = model.predict(final_input)[0]

        # Explanation (feature importance)
        importance = model.feature_importances_
        top_features = sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)[:3]

        result = "Approved ✅" if prediction == 1 else "Not Approved ❌"

        explanation = "\nTop Factors:\n"
        for f, imp in top_features:
            explanation += f"{f} ({round(imp,2)})\n"

        messagebox.showinfo("Result", f"Prediction: {result}\n{explanation}")

    except:
        messagebox.showerror("Error", "Please enter valid numbers")

# Button
tk.Button(root, text="Predict", command=predict, bg="green", fg="white").pack(pady=20)

root.mainloop()