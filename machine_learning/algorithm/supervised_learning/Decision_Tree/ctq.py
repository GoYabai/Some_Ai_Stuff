import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ==========================================
# STEP 1: INITIALIZE THE DATASET
# ==========================================
data = {
    'Diem_Tot_Nghiep': [12.0, 14.5, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0],
    'IELTS': ['Không', 'Không', 'Không', 'Không', 'Có', 'Không', 'Có', 'Không'],
    'Dan_Toc': ['Không', 'Có', 'Không', 'Có', 'Không', 'Không', 'Có', 'Không'],
    'Dau_Dai_Hoc': ['Không', 'Không', 'Không', 'Không', 'Có', 'Không', 'Có', 'Có']
}
df = pd.DataFrame(data)

# ==========================================
# STEP 2: DATA PREPROCESSING (ENCODING)
# ==========================================
# TODO 1: Create a dictionary to map 'Không' to 0 and 'Có' to 1
mapping_dict = {
    'Không':0,
    'Có':1 
}

# TODO 2: Apply the mapping to categorical columns to create new numerical columns
# Hint: Use pandas .map() function
df['IELTS_encoded'] = df['IELTS'].map(mapping_dict)
df['Dan_Toc_encoded'] = df['Dan_Toc'].map(mapping_dict)
df['Target'] = df['Dau_Dai_Hoc'].map(mapping_dict)

# ==========================================
# STEP 3: DEFINE FEATURES (X) AND TARGET (y)
# ==========================================
# TODO 3: Create feature matrix X and target vector y using the newly encoded columns
features = ['Diem_Tot_Nghiep', 'IELTS_encoded', 'Dan_Toc_encoded']
X = df[features]
y = df['Target']

# ==========================================
# STEP 4: BUILD AND TRAIN THE MODELS
# ==========================================
# TODO 4.1: Initialize and train a Decision Tree using 'entropy' (Information Gain)
# Hint: Set random_state=42 for reproducible results
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
# Fit the model
clf_entropy.fit(X, y)

# TODO 4.2: Initialize and train a Decision Tree using 'gini'
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
# Fit the model
clf_gini.fit(X, y)

# ==========================================
# STEP 5: PREDICT FOR A NEW STUDENT
# ==========================================
# The new student data: Diem_Tot_Nghiep = 21.0, IELTS = Không, Dan_Toc = Có
# TODO 5.1: Create a DataFrame for the new student using the encoded format
new_student = pd.DataFrame({
    'Diem_Tot_Nghiep': [21.0],
    'IELTS_encoded': ['0'],
    'Dan_Toc_encoded': ['1']
})

# TODO 5.2: Use the trained Gini model to predict the outcome
prediction = clf_gini.predict(new_student)
# Print the final result
result = "Có" if prediction[0] == 1 else "Không"
print(f"Prediction for the new student (Dau_Dai_Hoc): {result}")

# ==========================================
# OPTIONAL: VISUALIZE THE TREE
# ==========================================
# Uncomment the block below after completing the models to see the generated tree
plt.figure(figsize=(10, 5))
plot_tree(clf_gini, 
          feature_names=features, 
          class_names=['Truot', 'Dau'], 
          filled=True, 
          rounded=True)
plt.title("Decision Tree (Gini Impurity)")
plt.show()