#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns


# In[47]:


dataset = pd.read_csv('cities_magicbricks_rental_prices.csv')


# In[48]:


#rent is  target variable and other variables are assumed as features
threshold = dataset['rent'].median()


# In[49]:


#binary classification 
dataset['rent_category'] = (dataset['rent'] > threshold).astype(int) 


# In[50]:


features = dataset.drop(columns=['rent', 'rent_category'])  # Replace with feature column names
target = dataset['rent_category']


# In[51]:


dataset


# In[52]:


numerical_features = features.select_dtypes(include=['int64', 'float64']).columns


# In[53]:


categorical_features = features.select_dtypes(include=['object']).columns


# In[54]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# In[55]:


#Random Forests Algorithm


# In[56]:


clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# In[57]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# In[58]:


roc_auc_values = []
mean_fpr = np.linspace(0, 1, 100)
tprs = []


# In[59]:


results = []


# In[60]:


def calculate_metrics(tp, tn, fp, fn):
    """Calculates performance metrics."""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tss = (tp / (tp + fn)) - (fp / (fp + tn)) if (tp + fn > 0 and fp + tn > 0) else 0
    hss = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if (tp + fn > 0 and tp + fp > 0) else 0
    return accuracy, precision, recall, fpr, fnr, tss, hss


# In[61]:


fold = 1
plt.figure(figsize=(10, 8)) 
for train_idx, test_idx in skf.split(features, target):
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Predict
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy, precision, recall, fpr, fnr, tss, hss = calculate_metrics(tp, tn, fp, fn)
    
    fpr_roc, tpr_roc, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr_roc, tpr_roc)
    roc_auc_values.append(roc_auc)
    tprs.append(np.interp(mean_fpr, fpr_roc, tpr_roc))
    tprs[-1][0] = 0.0
    
    plt.plot(fpr_roc, tpr_roc, lw=2, alpha=0.6, label=f'Fold {fold} (AUC = {roc_auc:.2f})')

    
    # Append results
    results.append({
        'Fold': fold,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
        'FPR': fpr, 'FNR': fnr, 'TSS': tss, 'HSS': hss
    })
    fold += 1


# In[62]:


# Aggregate results

results_df = pd.DataFrame(results)
overall_metrics = results_df.mean(axis=0).to_dict()
overall_metrics['Fold'] = 'Average'


# In[63]:


# Final ROC curve
plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=f'Mean ROC (AUC = {mean_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest (10-Fold CV)')
plt.legend(loc='lower right')
plt.show()


# In[64]:


results_df = pd.concat([results_df, pd.DataFrame([overall_metrics])], ignore_index=True)


# In[65]:


print(results_df)
random_forests_metrics = results_df


# In[66]:


random_forests_metrics


# In[67]:


#KNN Algorithm


# In[68]:


# Load dataset (adjust file path and columns as needed)
data = pd.read_csv('cities_magicbricks_rental_prices.csv')

# Preprocessing
# Assuming 'rent' is the target variable. Adjust accordingly.
threshold = data['rent'].median()
data['rent_category'] = (data['rent'] > threshold).astype(int)  # Binary classification

features = data.drop(columns=['rent', 'rent_category'])  # Replace with feature column names
target = data['rent_category']

# Preprocessing pipeline
numerical_features = features.select_dtypes(include=['int64', 'float64']).columns
categorical_features = features.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Model pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))  # Using 5 nearest neighbors
])

# Stratified 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Metrics initialization
results = []


# Cross-validation loop
fold = 1
for train_idx, test_idx in skf.split(features, target):
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy, precision, recall, fpr, fnr, tss, hss = calculate_metrics(tp, tn, fp, fn)
    
    # Append results
    results.append({
        'Fold': fold,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
        'FPR': fpr, 'FNR': fnr, 'TSS': tss, 'HSS': hss
    })
    fold += 1
knn_probs = clf.predict_proba(X_test)[:, 1] 
# Aggregate results
results_df = pd.DataFrame(results)
overall_metrics = results_df.mean(axis=0).to_dict()
overall_metrics['Fold'] = 'Average'

# Append overall metrics
results_df = pd.concat([results_df, pd.DataFrame([overall_metrics])], ignore_index=True)

# Display results
print(results_df)


knn_metrics = results_df





# In[69]:


knn_metrics


# In[70]:


knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
knn_auc = auc(knn_fpr, knn_tpr)


# In[71]:


# Plot ROC curves
plt.figure(figsize=(10, 7))
plt.plot(knn_fpr, knn_tpr, label=f"KNN (AUC = {knn_auc:.2f})", color="green")

# Plot the random baseline
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[72]:


#GRU Algorithm


# In[73]:


# Load dataset (adjust file path and columns as needed)
data = pd.read_csv('cities_magicbricks_rental_prices.csv')

# Preprocessing
threshold = data['rent'].median()
data['rent_category'] = (data['rent'] > threshold).astype(int)  # Binary classification

features = data.drop(columns=['rent', 'rent_category'])  # Replace with feature column names
target = data['rent_category']

# Preprocessing pipeline
numerical_features = features.select_dtypes(include=['int64', 'float64']).columns
categorical_features = features.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

X = preprocessor.fit_transform(features)
y = target.values

# Reshape input for GRU (required to have 3 dimensions: samples, timesteps, features)
X = X.toarray() if hasattr(X, "toarray") else X  # Convert sparse matrix to dense if needed
X = X.reshape((X.shape[0], 1, X.shape[1]))  # Treat each sample as a "sequence" of 1 timestep

# Stratified 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Metrics initialization
results = []
mean_fpr = np.linspace(0, 1, 100)
tprs = []
roc_auc_values = []


# Cross-validation loop
fold = 1
plt.figure(figsize=(10, 8))  # Initialize the plot for the ROC curve
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Convert target to categorical for GRU
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)
    
    # Build GRU model
    model = Sequential([
        GRU(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train model
    model.fit(X_train, y_train_cat, epochs=50, batch_size=32, validation_data=(X_test, y_test_cat), callbacks=[early_stopping], verbose=0)
   
    # Predict probabilities for ROC curve
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy, precision, recall, fpr, fnr, tss, hss = calculate_metrics(tp, tn, fp, fn)
    
    # Append results
    results.append({
        'Fold': fold,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
        'FPR': fpr, 'FNR': fnr, 'TSS': tss, 'HSS': hss
    })
    
    # Calculate ROC AUC
    fpr_roc, tpr_roc, _ = roc_curve(y_test, y_pred_prob[:, 1])
    roc_auc = auc(fpr_roc, tpr_roc)
    roc_auc_values.append(roc_auc)
    tprs.append(np.interp(mean_fpr, fpr_roc, tpr_roc))
    tprs[-1][0] = 0.0
    
    plt.plot(fpr_roc, tpr_roc, lw=2, alpha=0.6, label=f'Fold {fold} (AUC = {roc_auc:.2f})')
    fold += 1

# Final ROC curve
plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=f'Mean ROC (AUC = {mean_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for GRU (10-Fold CV)')
plt.legend(loc='lower right')

plt.show()

# Aggregate results
results_df = pd.DataFrame(results)
overall_metrics = results_df.mean(axis=0).to_dict()
overall_metrics['Fold'] = 'Average'

# Append overall metrics
results_df = pd.concat([results_df, pd.DataFrame([overall_metrics])], ignore_index=True)

# Save results

# Display results
print(results_df)


# In[74]:


Gru_metrics = results_df


# In[75]:


Gru_metrics


# In[82]:


# Adding an algorithm identifier for each data frame
Gru_metrics['Algorithm'] = 'GRU'
knn_metrics['Algorithm'] = 'KNN'
random_forests_metrics['Algorithm'] = 'Random Forest'

# Combine all metrics for comparison
all_metrics = pd.concat([Gru_metrics, knn_metrics, random_forests_metrics], ignore_index=True)

# Select metrics for visualization
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'TSS', 'HSS']

# Visualization
plt.figure(figsize=(16, 10))
for i, metric in enumerate(metrics_to_plot, start=1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=all_metrics, x='Algorithm', y=metric, palette='Set2')
    plt.title(f'{metric} Comparison')
    plt.ylabel(metric)
    plt.xlabel('Algorithm')

plt.tight_layout()
 
plt.show()

# Line plot for average performance per algorithm
average_metrics = all_metrics.groupby('Algorithm')[metrics_to_plot].mean().reset_index()

plt.figure(figsize=(10, 6))
for metric in metrics_to_plot:
    plt.plot(average_metrics['Algorithm'], average_metrics[metric], marker='o', label=metric)

plt.title('Average Performance Comparison')
plt.ylabel('Metric Value')
plt.xlabel('Algorithm')
plt.legend(title='Metrics', loc='lower right')
plt.grid()
plt.show()

