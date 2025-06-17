import pandas as pd
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Configure pandas to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv('./social_media_vs_productivity.csv')

# ETAPA 0 - Dados preliminares do conjunto de dados
print("\nInformações básicas do dataset:")
print(data.info())

# ETAPA 1 - Limpeza dos dados e análise inicial
# Remove null values from actual_productivity_score column
data = data.dropna(subset=['actual_productivity_score'])
data = data.drop(['age','social_platform_preference', 'uses_focus_apps', 'has_digital_wellbeing_enabled', 'breaks_during_work'], axis=1)

# Identificando colunas numéricas (contínuas) e categóricas
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
int_columns = data.select_dtypes(include=['int64']).columns.tolist()
categorical_columns = ['job_type']

# Identify and remove outliers using IQR method
for column in continuous_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[~((data[column] < lower_bound) | (data[column] > upper_bound))]

print("\nInformações do dataset após remoção de outliers:")
print(data.info())

# Verificando valores faltantes em todas as colunas após remoção
missing_values = data.isnull().sum()
missing_percentage = (data.isnull().sum() / len(data)) * 100

print("\nValores faltantes em todas as colunas após remoção:")
for col, missing in missing_values.items():
    if missing > 0:
        print(f"{col}: {missing} valores faltantes ({missing_percentage[col]:.2f}%)")
        print()

# Create numeric pipeline for imputation
float_imputer = SimpleImputer(strategy='mean')
data[continuous_columns] = float_imputer.fit_transform(data[continuous_columns])
int_imputer = SimpleImputer(strategy='median')
data[int_columns] = float_imputer.fit_transform(data[int_columns])

# Create categorical pipeline for job_type
onehot = OneHotEncoder(sparse_output=False, drop='first')
job_type_encoded = onehot.fit_transform(data[categorical_columns])

print("\nDados após imputação:")
print(data.info())

# ETAPA 2 - Tratamento e descrição dos de dados

# Transformando actual_productivity_score em categorias
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
data['productivity_category'] = kbd.fit_transform(data[['actual_productivity_score']])
data['productivity_category'] = data['productivity_category'].astype(int)

# Mostrando a distribuição das categorias
print("\nDistribuição das categorias de produtividade:")
print(data['productivity_category'].value_counts().sort_index())
print("\nEstatísticas por categoria:")
print(data.groupby('productivity_category')['actual_productivity_score'].describe())

# ETAPA 3 - Treinamento dos modelos preditivos

# Separando features e target
features = data.drop(['productivity_category', 'actual_productivity_score'], axis=1)
target = data['productivity_category']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Criando pipeline de pré-processamento
numeric_features = features.select_dtypes(include=['float64', 'int64']).columns
categorical_features = features.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Treinando DecisionTreeClassifier
dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

dt_pipeline.fit(X_train, y_train)
dt_score = dt_pipeline.score(X_test, y_test)
print("\nDecision Tree Accuracy:", dt_score)


# Treinando MLPClassifier
mlp_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(random_state=42, max_iter=1000))
])

mlp_pipeline.fit(X_train, y_train)
mlp_score = mlp_pipeline.score(X_test, y_test)
print("MLP Classifier Accuracy:", mlp_score)

# Calculando e mostrando as métricas de avaliação para Decision Tree
dt_pred = dt_pipeline.predict(X_test)
print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred))

# Calculando e mostrando as métricas de avaliação para MLP
mlp_pred = mlp_pipeline.predict(X_test)
print("\nMLP Classifier Confusion Matrix:")
print(confusion_matrix(y_test, mlp_pred))
print("\nMLP Classifier Classification Report:")
print(classification_report(y_test, mlp_pred))
