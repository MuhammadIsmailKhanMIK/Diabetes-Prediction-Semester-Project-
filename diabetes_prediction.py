import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from experta import KnowledgeEngine, Rule, Fact, P
import tkinter as tk
from tkinter import messagebox

# Load Pima Indians Diabetes Dataset from a local file
file_path = "diabetes.csv"
data = pd.read_csv(file_path)

# Separate features (X) and target (y) from the dataset
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Knowledge Base for Reasoning
class DiabetesKnowledgeEngine(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.reasons = []  # List to store reasoning explanations

    @Rule(Fact(glucose=P(lambda x: x > 140)))
    def high_glucose(self):
        self.reasons.append("High glucose level detected, risk of diabetes.")

    @Rule(Fact(bmi=P(lambda x: x > 30)))
    def high_bmi(self):
        self.reasons.append("BMI indicates overweight, increasing diabetes risk.")

    @Rule(Fact(age=P(lambda x: x > 45)))
    def high_age(self):
        self.reasons.append("Age is a risk factor for diabetes.")

    @Rule(Fact(family_history=True))
    def family_history_with_risk(self):
        self.reasons.append("Family history of diabetes detected. Consult a doctor for personalized advice.")

    @Rule(Fact(DiabetesPedigreeFunction=P(lambda x: x > 0.5)))  # Detect family history if DPF > 0.5
    def family_history(self):
        self.reasons.append("Family history of diabetes detected.")

    @Rule(Fact(glucose=P(lambda x: x > 140)))
    def provide_preventive_measures(self):
        self.reasons.append("It is recommended to monitor your blood sugar levels, maintain a balanced diet, and consult a doctor for personalized advice.")

    @Rule(Fact(bmi=P(lambda x: x > 30)))
    def provide_preventive_measures_bmi(self):
        self.reasons.append("Maintaining a healthy BMI is important. Consider regular physical activity and consult a doctor for a tailored fitness plan.")

# GUI for User Input and Prediction
class HealthGuardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HealthGuard: Diabetes Prediction")
        self.root.geometry("600x800")  # Increased height for a larger window
        self.root.configure(bg='#2e2e2e')

        self.fields = {}
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Diabetes Prediction", font=("Segoe UI", 18, "bold", "underline"), bg="#2e2e2e", fg="#ffffff").pack(pady=20)

        # Gender selection
        tk.Label(self.root, text="Select Gender:", font=("Segoe UI", 12), bg="#2e2e2e", fg="#ffffff").pack(pady=5)
        self.gender_var = tk.StringVar(value="female")
        gender_frame = tk.Frame(self.root, bg="#2e2e2e")
        gender_frame.pack(pady=10)
        
        tk.Radiobutton(gender_frame, text="Male", variable=self.gender_var, value="male", 
                       font=("Segoe UI", 12), bg="#2e2e2e", fg="#ffffff", selectcolor="#404040", 
                       command=self.display_fields).pack(side=tk.LEFT, padx=20)
        tk.Radiobutton(gender_frame, text="Female", variable=self.gender_var, value="female", 
                       font=("Segoe UI", 12), bg="#2e2e2e", fg="#ffffff", selectcolor="#404040", 
                       command=self.display_fields).pack(side=tk.LEFT, padx=20)

        # Form frame for inputs
        self.form_frame = tk.Frame(self.root, bg="#2e2e2e")
        self.form_frame.pack(pady=10, fill="x")

        self.fields = {
            "Pregnancies": "Number of Pregnancies:",
            "Glucose": "Plasma Glucose Concentration:",
            "BloodPressure": "Diastolic Blood Pressure (mm Hg):",
            "SkinThickness": "Triceps Skinfold Thickness (mm):",
            "Insulin": "2-hour Serum Insulin (mu U/ml):",
            "BMI": "Body Mass Index (BMI):",
            "DiabetesPedigreeFunction": "Diabetes Pedigree Function:",
            "Age": "Age:"
        }

        self.entries = {}
        self.display_fields()

        # Prediction button
        tk.Button(self.root, text="Predict", font=("Segoe UI", 12, "bold"), bg="#0052cc", fg="#ffffff", command=self.predict_disease).pack(pady=20)

        # Prediction result and reasoning output
        self.result_label = tk.Label(self.root, text="", font=("Segoe UI", 14, "bold"), bg="#2e2e2e", fg="#ffffff")
        self.result_label.pack(pady=10)

        # Reasoning output in a scrollable Text widget with increased height
        self.reasoning_label = tk.Label(self.root, text="Reasoning Output", font=("Segoe UI", 14, "bold"), bg="#2e2e2e", fg="#ffffff")
        self.reasoning_label.pack(pady=10)

        self.reasoning_text = tk.Text(self.root, height=15, bg="#404040", fg="#ffffff", wrap="word", font=("Segoe UI", 12))
        self.reasoning_text.pack(pady=10, fill="x")  # Increased height
        self.reasoning_text.config(state=tk.DISABLED)  # Initially disable text entry

    def display_fields(self, *args):
        """Display appropriate input fields based on selected gender."""
        for widget in self.form_frame.winfo_children():
            widget.destroy()

        gender = self.gender_var.get()
        fields_to_display = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

        if gender == "female":
            fields_to_display.insert(0, "Pregnancies")

        self.entries = {}
        for field in fields_to_display:
            frame = tk.Frame(self.form_frame, bg="#2e2e2e")
            frame.pack(pady=5, fill="x")
            tk.Label(frame, text=self.fields[field], font=("Segoe UI", 12), bg="#2e2e2e", fg="#ffffff").pack(side=tk.LEFT, padx=10)
            entry = tk.Entry(frame, bg="#404040", fg="#ffffff", font=("Segoe UI", 12))
            entry.pack(side=tk.RIGHT, padx=10, fill="x", expand=True)
            self.entries[field] = entry

    def predict_disease(self):
        try:
            # Gather all inputs
            features = []
            gender = self.gender_var.get()

            for key, entry in self.entries.items():
                if key == "Pregnancies" and gender == "male":
                    features.append(0)  # Default for males
                else:
                    features.append(float(entry.get() or 0))  # Default to 0 if empty

            # Align features with model's expected input
            if "Pregnancies" not in self.entries:
                features.insert(0, 0)  # Add a placeholder for "Pregnancies" for male

            input_data = pd.DataFrame([features], columns=X.columns)
            input_data = scaler.transform(input_data)
            prediction = model.predict(input_data)[0]

            # Display Prediction Result
            self.result_label.config(text=f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")

            # Reasoning
            engine = DiabetesKnowledgeEngine()
            engine.reset()
            engine.declare(Fact(glucose=features[1]), Fact(bmi=features[5]), Fact(age=features[-1]), Fact(family_history=(features[6] > 0.5)), Fact(DiabetesPedigreeFunction=features[6]))
            engine.run()

            # Display Reasoning
            reasoning = "\n".join(engine.reasons) or "No significant risks identified."
            self.update_reasoning_output(reasoning)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric data.")

    def update_reasoning_output(self, reasoning):
        """Update the reasoning text area with the explanation."""
        self.reasoning_text.config(state=tk.NORMAL)  # Enable text area to update
        self.reasoning_text.delete("1.0", tk.END)  # Clear previous text
        self.reasoning_text.insert(tk.END, reasoning)  # Insert new reasoning
        self.reasoning_text.config(state=tk.DISABLED)  # Disable editing

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = HealthGuardApp(root)
    root.mainloop()