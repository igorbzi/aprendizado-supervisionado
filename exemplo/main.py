import pandas as pd
import dtreeviz
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Configure pandas to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Stroke Prediction Dataset
# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data

# Carregar o dataset com biblioteca pandas
data = pd.read_csv('./healthcare-dataset-stroke-data.csv')

# ETAPA 0 - Dados preliminares do conjunto de dados
print("\nInformações básicas do dataset:")
print(data.info())

# ETAPA 1 - Limpeza dos dados e análise inicial
# Remove ID column as it's not relevant for prediction
data.drop('id', axis=1, inplace=True)
# Remove sample with gender Other (Apenas uma amostra, portanto é irrelevante)
data = data[data['gender'] != 'Other']
data.reset_index(drop=True, inplace=True)

# ETAPA 2 - Tratamento e descrição dos de dados
print("\nAtributos contínuos:")

print("- age (idade)")
print("- avg_glucose_level (nível médio de glicose)")
print("- bmi (índice de massa corporal)")
# Calcular estatísticas para atributos contínuos
continuous_attrs = ['age', 'avg_glucose_level', 'bmi']
stats = data[continuous_attrs].describe()
print("\nEstatísticas dos atributos contínuos:")
print(stats)
# Handle missing values in BMI usando a média
imputer = SimpleImputer(strategy='mean')
data['bmi'] = imputer.fit_transform(data[['bmi']])
# Calcular estatísticas para atributos contínuos
print("\nApós imputação em bmi com Null")
stats = data[continuous_attrs].describe()
print(stats)

print("\nAtributos categóricos ordinais:")

print("- smoking_status (status de fumante: never smoked -> formerly smoked -> smokes)")
print("\nValores únicos em smoking_status:")
print(data['smoking_status'].value_counts())
# Calculate likelihood based on relevant features
def calculate_smoking_likelihood(row):
    # Get similar cases based on age, gender, and other relevant features
    similar_cases = data[
        (abs(data['age'] - row['age']) <= 22) &  # Considerado o desvio padrão do conjunto como distância
        (abs(data['avg_glucose_level'] - row['avg_glucose_level']) <= 45) &
        (abs(data['bmi'] - row['bmi']) <= 8) &
        (data['gender'] == row['gender']) &
        (data['hypertension'] == row['hypertension']) &
        (data['heart_disease'] == row['heart_disease']) &
        (data['ever_married'] == row['ever_married']) &
        (data['work_type'] == row['work_type']) &
        (data['Residence_type'] == row['Residence_type']) &
        (data['smoking_status'] != 'Unknown')
        ]

    if len(similar_cases) > 0:
        # Return most common smoking status among similar cases
        return similar_cases['smoking_status'].mode()[0]
    else:
        # If no similar cases found, return overall mode
        return data[data['smoking_status'] != 'Unknown']['smoking_status'].mode()[0]
# Apply likelihood-based imputation for Unknown smoking status
unknown_mask = data['smoking_status'] == 'Unknown'
data.loc[unknown_mask, 'smoking_status'] = data[unknown_mask].apply(calculate_smoking_likelihood, axis=1)
# Map smoking status to numerical values
mapping = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}
data['smoking_status'] = data['smoking_status'].map(mapping)
print("\nApós imputação em Uknown e conversão para valores numéricos")
print("- smoking_status (status de fumante: 0 -> 1 -> 2)")
print("Valores únicos em smoking_status:")
print(data['smoking_status'].value_counts())

print("\nAtributos categóricos binários:")
print("- gender (gênero)")
print("- hypertension (hipertensão)")
print("- heart_disease (doença cardíaca)")
print("- ever_married (já foi casado)")
print("- Residence_type (tipo de residência)")
print("- stroke (derrame - atributo alvo)")

print("\nValores únicos em gender:")
print(data['gender'].value_counts())
mapping = {'Female': 0, 'Male': 1}
data['gender'] = data['gender'].map(mapping)
print("\nValores únicos em hypertension:")
print(data['hypertension'].value_counts())
print("\nValores únicos em heart_disease:")
print(data['heart_disease'].value_counts())
print("\nValores únicos em ever_married:")
print(data['ever_married'].value_counts())
mapping = {'No': 0, 'Yes': 1}
data['ever_married'] = data['ever_married'].map(mapping)
print("\nValores únicos em Residence_type:")
print(data['Residence_type'].value_counts())
# Map smoking status to numerical values
mapping = {'Urban': 0, 'Rural': 1}
data['Residence_type'] = data['Residence_type'].map(mapping)
print("\nValores únicos em stroke:")
print(data['stroke'].value_counts())

print("\nAtributos categóricos simples:")
print("- work_type (tipo de trabalho)")
print("\nValores únicos em work_type:")
print(data['work_type'].value_counts())
# One-hot encoding para work_type
encoder = OneHotEncoder(sparse_output=False)
work_type_encoded = encoder.fit_transform(data[['work_type']])
work_type_encoded_df = pd.DataFrame(work_type_encoded, columns=encoder.get_feature_names_out(['work_type']))
data = pd.concat([data, work_type_encoded_df], axis=1)
data.drop('work_type', axis=1, inplace=True)

print("\nEstatísticas de todos os dados após tratamentos:")
print(data.info())
print(data.describe())

# ETAPA 3 - Treinamento dos modelos preditivos
# Prepare data for model training
X = data.drop('stroke', axis=1)
y = data['stroke']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Árvores de Decisão
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 0.01, 0.05, 0.1],
    'min_impurity_decrease': [0.0001],
    'class_weight': ['balanced']
}

dt = DecisionTreeClassifier()
grid_search = GridSearchCV(dt, param_grid, cv=20, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nMelhores parâmetros encontrados:")
print(grid_search.best_params_)
print("\nMelhor score de validação cruzada:")
cv_results_df = pd.DataFrame(grid_search.cv_results_)
print(cv_results_df[['param_criterion', 'param_max_depth', 'param_min_samples_split', 'mean_test_score', 'std_test_score', 'rank_test_score']])

model = grid_search.best_estimator_

y_pred = model.predict(X_train)
print("\nÁrvore de Decisão - Treino")
print("\nClassification Report:")
print(classification_report(y_train, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_pred))

y_pred = model.predict(X_test)
print("\nÁrvore de Decisão - Teste")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualização das árvores
if False:
    viz_model = dtreeviz.model(model,
        X_train=X_train,
        y_train=y_train,
        feature_names=X.columns.tolist(),
        target_name="Stroke",
        class_names=['No Stroke', 'Stroke']
    )
    v = viz_model.view()
    v.save("decision_tree.svg")

    fig = plt.figure(figsize=(200,200))
    _ = tree.plot_tree(model,
                       feature_names=X.columns.tolist(),
                       class_names=['No Stroke', 'Stroke'],
                       filled=True)
    fig.savefig("decistion_tree.png")

# Redes Neurais Artificiais
# Normalize the features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train neural network model
nn_model = MLPClassifier(
    hidden_layer_sizes=(50, 50, 50, 50, 50, 50, 50),
    activation='relu',
    solver='adam',
    max_iter=100000,
)
nn_model.fit(X_train_scaled, y_train)

y_pred = nn_model.predict(X_train_scaled)
print("\nRede Neural - Treino")
print("\nClassification Report:")
print(classification_report(y_train, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_pred))

y_pred = nn_model.predict(X_test_scaled)
print("\nRede Neural - Teste")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
