#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Sample data for illustration
np.random.seed(42)
data = {
    'juror_id': range(1, 101),
    'prior_convictions': np.random.randint(0, 2, 100),  # Binary variable representing prior convictions
    'neighborhood_crime_rate': np.random.uniform(0, 1, 100),  # Continuous variable representing crime rate
    'favor_prosecution': np.random.randint(0, 2, 100)  # Binary target variable representing likelihood to favor prosecution
}
df = pd.DataFrame(data)

# Features and target variable
X = df[['prior_convictions', 'neighborhood_crime_rate']]
y = df['favor_prosecution']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Train Logistic Regression model
logit = LogisticRegression()
logit.fit(X_train, y_train)

# Initialize lists to store accepted jurors for prosecution and defense
prosecution_accepted_rf = []
defense_accepted_rf = []
prosecution_accepted_logit = []
defense_accepted_logit = []
prosecution_accepted_ensemble = []
defense_accepted_ensemble = []

# Simulate jury selection process for Random Forest
jury_pool = list(df['juror_id'])
for _ in range(12):
    juror_to_remove = np.random.choice(jury_pool)
    prosecution_accepted_rf.append(juror_to_remove)
    jury_pool.remove(juror_to_remove)

    juror_to_remove = np.random.choice(jury_pool)
    defense_accepted_rf.append(juror_to_remove)
    jury_pool.remove(juror_to_remove)

# Simulate jury selection process for Logistic Regression
jury_pool = list(df['juror_id'])
for _ in range(12):
    juror_to_remove = np.random.choice(jury_pool)
    prosecution_accepted_logit.append(juror_to_remove)
    jury_pool.remove(juror_to_remove)

    juror_to_remove = np.random.choice(jury_pool)
    defense_accepted_logit.append(juror_to_remove)
    jury_pool.remove(juror_to_remove)

# Simulate jury selection process for Ensemble
jury_pool = list(df['juror_id'])
for _ in range(12):
    juror_to_remove = np.random.choice(jury_pool)
    prosecution_accepted_ensemble.append(juror_to_remove)
    jury_pool.remove(juror_to_remove)

    juror_to_remove = np.random.choice(jury_pool)
    defense_accepted_ensemble.append(juror_to_remove)
    jury_pool.remove(juror_to_remove)

# Evaluate models
y_pred_rf = rf.predict(X_test)
y_pred_logit = logit.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_logit = accuracy_score(y_test, y_pred_logit)

print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Logistic Regression Accuracy: {accuracy_logit:.4f}")

# Ensemble model
ensemble = VotingClassifier(estimators=[('rf', rf), ('logit', logit)], voting='soft')
ensemble.fit(X_train, y_train)

y_pred_ensemble = ensemble.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)

print(f"Ensemble Accuracy: {accuracy_ensemble:.4f}")

print("\nRandom Forest:")
print("Prosecution's Accepted Jurors:", prosecution_accepted_rf)
print("Defense's Accepted Jurors:", defense_accepted_rf)

print("\nLogistic Regression:")
print("Prosecution's Accepted Jurors:", prosecution_accepted_logit)
print("Defense's Accepted Jurors:", defense_accepted_logit)

print("\nEnsemble:")
print("Prosecution's Accepted Jurors:", prosecution_accepted_ensemble)
print("Defense's Accepted Jurors:", defense_accepted_ensemble)


# In[ ]:




