import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from scipy.stats import spearmanr



data_x = pd.read_csv('Data\Data_X.csv')
data_y = pd.read_csv('Data\Data_Y.csv')
data_new_x = pd.read_csv('Data\DataNew_X.csv')
# Convertir les valeurs de la colonne 'COUNTRY' en valeurs numériques
data_x['COUNTRY'] = data_x['COUNTRY'].replace({'FR': 0, 'DE': 1})

#Données manquantes
missing_values_x = data_x.isnull().sum()
missing_values_y = data_y.isnull().sum()
missing_values_new_x = data_new_x.isnull().sum()
print("Missing values in Data X:\n", missing_values_x)
print("Missing values in Data Y:\n", missing_values_y)
print("Missing values in DataNew X:\n", missing_values_new_x)

#attributs comparables
data_x.describe()
#Normalisation des données
scaler = MinMaxScaler()
data_x_scaled = pd.DataFrame(scaler.fit_transform(data_x.drop(columns=['ID', 'DAY_ID', 'COUNTRY'])), columns=data_x.columns[3:])

#Suppression des attributs non pertinents (blancs)
# Remplacer les valeurs manquantes par la moyenne de la colonne
data_x_filled = data_x.fillna(data_x.mean())

# Supprimer les lignes avec des valeurs manquantes
data_x_dropped = data_x.dropna()

#Fractionnez les données en sous-ensembles d'apprentissage et de test
# Fusionnez Data X et Data Y sur la colonne 'ID'
# Fusionnez Data X et Data Y sur la colonne 'ID'
merged_data = pd.merge(data_x_filled, data_y, on='ID')
X = merged_data.drop(columns=['ID', 'DAY_ID', 'COUNTRY', 'TARGET'])
y = merged_data['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#tri des données par jour 
data_x_sorted = data_x.sort_values(by=['DAY_ID'])

# Sélectionnez les données uniquement pour la France
#data_x_fr = data_x[data_x['COUNTRY'] == 'FR']
#fusion des données 
#merged_data = pd.merge(data_x, data_y, on='ID')



# Analyse exploratoire des données (EDA)
# Aperçu des variables
print(data_x.dtypes)

# Histogrammes
'''numerical_columns = data_x.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data_x[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()

# Diagrammes en boîte
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='COUNTRY', y=col, data=data_x)
    plt.title(f'Boxplot of {col} by Country')
    plt.show()'''

# Matrice de corrélation
corr_matrix = data_x.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()




# Créez une liste des modèles que vous souhaitez entraîner
models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('K-Nearest Neighbors', KNeighborsRegressor()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor())
]

# Créez une fonction pour entraîner et évaluer un modèle
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spearman_corr = spearmanr(y_test, y_pred)[0]

    return mse, r2, spearman_corr

# Entraînez et évaluez chaque modèle
results = []
for name, model in models:
    mse, r2, spearman_corr = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
    results.append((name, mse, r2, spearman_corr))
    print(f"{name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  Spearman Correlation: {spearman_corr:.4f}\n")

# Comparez les performances des modèles
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R2', 'Spearman Correlation'])
results_df = results_df.sort_values(by='R2', ascending=False)
print("Performance ranking:")
print(results_df)


import matplotlib.pyplot as plt

# Créez un graphique à barres pour comparer les performances des modèles
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
x = np.arange(len(results_df))

ax.bar(x - bar_width, results_df['MSE'], width=bar_width, label='MSE')
ax.bar(x, results_df['R2'], width=bar_width, label='R2')
ax.bar(x + bar_width, results_df['Spearman Correlation'], width=bar_width, label='Spearman Correlation')

ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=45)
ax.legend()

plt.title("Performance Comparison")
plt.show()