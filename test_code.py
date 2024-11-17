import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import simpledialog, messagebox

# Load dataset from CSV file
df = pd.read_csv('autism_dataset.csv')

# Feature columns
features = [
    'Social_Interaction_Avoids_Eye_Contact', 'Social_Interaction_Uninterested',
    'Communication_Trouble_Conversations', 'Communication_Few_Gestures',
    'Repetitive_Behaviors_Repetitive_Movements', 'Repetitive_Behaviors_Distress_With_Change',
    'Sensory_Sensitivities_Sensitive_Noises_Textures', 'Sensory_Sensitivities_Strong_Reaction_Lights_Sounds',
    'Developmental_Milestones_Delay', 'Developmental_Milestones_Difficulty_Milestones',
    'Emotional_Regulation_Tantrums', 'Emotional_Regulation_Limited_Emotions',
    'Behavioral_Patterns_Focused_Interests', 'Behavioral_Patterns_Unusual_Behaviors',
    'Response_to_Treatment_Improvement', 'Response_to_Treatment_Progress_With_Support', 
    'age', 'gender', 'ethnicity', 'jaundice'
]

# Target variable
target = 'autism'

# Split data into features and target
X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Tkinter UI setup
root = tk.Tk()
root.title("ASD Prediction")

# Function to get user input with percentage
def get_yes_no_input(question):
    answer = messagebox.askquestion("Question", question)
    if answer == 'yes':
        percentage = simpledialog.askinteger("Percentage", "Enter the percentage (1-100):", minvalue=1, maxvalue=100)
        return percentage / 100  # Normalize percentage to a value between 0 and 1
    else:
        return 0  # If "No", return 0 (0%)

# Function to predict autism based on user input
def predict_autism_rf():
    try:
        user_data = []

        # Collect user inputs using Yes/No questions
        user_data.append(get_yes_no_input("Social Interaction: Avoids eye contact?"))
        user_data.append(get_yes_no_input("Social Interaction: Uninterested in social interactions?"))
        user_data.append(get_yes_no_input("Communication: Trouble starting or maintaining conversations?"))
        user_data.append(get_yes_no_input("Communication: Uses few or no gestures?"))
        user_data.append(get_yes_no_input("Repetitive Behaviors: Engages in repetitive movements?"))
        user_data.append(get_yes_no_input("Repetitive Behaviors: Distress with changes in routine?"))
        user_data.append(get_yes_no_input("Sensory Sensitivities: Sensitive to noises or textures?"))
        user_data.append(get_yes_no_input("Sensory Sensitivities: Strong reaction to lights or sounds?"))
        user_data.append(get_yes_no_input("Developmental Milestones: Delayed in speaking or walking?"))
        user_data.append(get_yes_no_input("Developmental Milestones: Difficulty meeting milestones?"))
        user_data.append(get_yes_no_input("Emotional Regulation: Frequent tantrums or emotional outbursts?"))
        user_data.append(get_yes_no_input("Emotional Regulation: Limited range of emotions?"))
        user_data.append(get_yes_no_input("Behavioral Patterns: Focused on specific interests or objects?"))
        user_data.append(get_yes_no_input("Behavioral Patterns: Exhibits unusual behaviors in social settings?"))
        user_data.append(get_yes_no_input("Response to Treatment: Shown improvement with therapy or interventions?"))
        user_data.append(get_yes_no_input("Response to Treatment: Made progress with support?"))

        # Fixed user input (age, gender, ethnicity, jaundice)
        age = simpledialog.askinteger("Input", "Age:", minvalue=1)
        gender = simpledialog.askinteger("Input", "Gender (1 for male, 0 for female):", minvalue=0, maxvalue=1)
        ethnicity = simpledialog.askinteger("Input", "Ethnicity (1: Asian, 2: Black, 3: White, etc.):", minvalue=1)
        jaundice = simpledialog.askinteger("Input", "Jaundice (1 for yes, 0 for no):", minvalue=0, maxvalue=1)

        # Append fixed inputs to user_data
        user_data.extend([age, gender, ethnicity, jaundice])

        # Create DataFrame for user input
        user_data_df = pd.DataFrame([user_data], columns=features)

        # Scale the input data
        user_data_scaled = scaler.transform(user_data_df)

        # Make prediction using Random Forest
        prediction = rf_model.predict(user_data_scaled)
        probability = rf_model.predict_proba(user_data_scaled)[0][1] * 100  # Probability for class '1'

        # Calculate the total score and determine result based on majority
        total_yes_count = sum(1 for score in user_data[:16] if score > 0)  # Count how many symptoms have 'Yes' responses

        # Show result in a message box
        if total_yes_count > 8:  # More than half
            result_message = (
                f"The person is predicted to have Autism Spectrum Disorder with a model confidence of {probability:.2f}%. "
                f"The total percentage of positive symptoms is approximately {sum(user_data[:16]) / 16 * 100:.2f}%."
            )
        else:
            result_message = (
                "The person is predicted not to have Autism Spectrum Disorder."
            )

        messagebox.showinfo("Prediction Result", result_message)

    except Exception as e:
        messagebox.showerror("Input Error", f"An error occurred: {str(e)}")

# Button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict_autism_rf)
predict_button.pack()

root.mainloop()
