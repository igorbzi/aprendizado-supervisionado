import pandas as pd
import dtreeviz
import seaborn as sns
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


# Função para análise completa de outliers
def analyze_outliers(df, column):
    # Estatísticas
    print(f"\nAnálise de outliers para: {column}")
    print(df[column].describe())
    
    # Detecção pelo método IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Número de outliers detectados: {len(outliers)}")
    print(f"Porcentagem do total: {len(outliers)/len(df)*100:.2f}%")
    
    return outliers


# Configure pandas to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv('./social_media_vs_productivity.csv')
#estresse com base em horas de tela e de sono

# ETAPA 0 - Dados preliminares do conjunto de dados
print("\nInformações básicas do dataset:")
print(data.info())

# ETAPA 1 - Limpeza dos dados e análise inicial
data.drop('social_platform_preference', axis=1, inplace=True)


# ETAPA 2 - Tratamento e descrição dos de dados
print("\nAtributos contínuos:")

# Calcular estatísticas para atributos contínuos
continuous_attrs = ['age', 'daily_social_media_time', 'perceived_productivity_score', 'actual_productivity_score', 'stress_level', 'sleep_hours', 'screen_time_before_sleep', 'weekly_offline_hours', 'job_satisfaction_score']
stats = data[continuous_attrs].describe()
print("\nEstatísticas dos atributos contínuos:")
print(stats)

# Analisar variáveis chave
sleep_outliers = analyze_outliers(data, 'sleep_hours')
social_outliers = analyze_outliers(data, 'daily_social_media_time')
productivity_outliers = analyze_outliers(data, 'actual_productivity_score')


# ETAPA 3 - Treinamento dos modelos preditivos

""" 
#Árvores de Decisão
# Prepare data for model training
X = data.drop('stroke', axis=1)
y = data['stroke']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

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
"""