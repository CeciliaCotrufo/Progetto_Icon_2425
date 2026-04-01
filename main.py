import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from owlready2 import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

print("Caricamento Ontologia e Dataset (Campione di 5000 istanze)")
onto = get_ontology("file://ontologia/ontologia_funghi.owl").load()
df = pd.read_csv('database/mushrooms.csv')
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

print("Caricamento Dati nell'Ontologia")
with onto:
    for index, row in df.iterrows():
        nome_fungo = f"Fungo_{index}"
        nuovo_fungo = onto.Fungo(nome_fungo)
        nuovo_fungo.ha_odore.append(row['odor'])
        nuovo_fungo.colore_spore.append(row['spore-print-color'])

print("Avvio HermiT")
sync_reasoner()

funghi_tossici_inferiti = [f.name for f in onto.Tossico_Biologico.instances()]
df['Deduzione_Ontologia'] = df.apply(lambda r: 1 if f"Fungo_{r.name}" in funghi_tossici_inferiti else 0, axis=1)

print("Feature Selection per il Raccoglitore inesperto")
le = LabelEncoder()
feature_visive = ['cap-shape', 'cap-surface', 'cap-color', 'gill-spacing', 'stalk-shape', 'habitat', 'population']

for col in feature_visive:
    df[col] = le.fit_transform(df[col])

X_base = df[feature_visive]
X_arricchito = df[feature_visive + ['Deduzione_Ontologia']]
y = le.fit_transform(df['class'])

print("Addestramento Multi-Modello con NESTED CROSS-VALIDATION\n")
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

modelli_nested = {
    "Decision Tree": (DecisionTreeClassifier(random_state=42),
                      {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}),
    "Random Forest": (RandomForestClassifier(random_state=42), {'criterion': ['gini', 'entropy'], 'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5],
                       'min_samples_leaf': [1, 2]}),
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear', 'lbfgs'],
                             'class_weight': [None, 'balanced']}),
    "Support Vector Machine": (SVC(kernel='linear', probability=True, random_state=42), {'C': [0.1, 1.0, 10.0]})
}

predizioni_modelli = {}
parametri_migliori = {}

print("RISULTATI FINALI COMPARATIVI (NESTED 10-FOLD CV)")

for nome, (modello, griglia_parametri) in modelli_nested.items():
    print(f"Calcolo in corso per {nome}")
    clf_ottimizzato = GridSearchCV(estimator=modello, param_grid=griglia_parametri, cv=cv_inner, scoring='accuracy')

    score_base = cross_val_score(clf_ottimizzato, X_base, y, cv=cv_outer, scoring='accuracy', n_jobs=-1)
    score_arr = cross_val_score(clf_ottimizzato, X_arricchito, y, cv=cv_outer, scoring='accuracy', n_jobs=-1)

    y_pred = cross_val_predict(clf_ottimizzato, X_arricchito, y, cv=cv_outer, n_jobs=-1)
    predizioni_modelli[nome] = y_pred

    clf_ottimizzato.fit(X_arricchito, y)
    parametri_migliori[nome] = clf_ottimizzato.best_params_

    print(f" {nome} ")
    print(f" BASE:       {score_base.mean() * 100:.2f}% (Std: ±{score_base.std() * 100:.2f}%)")
    print(f" ARRICCHITO: {score_arr.mean() * 100:.2f}% (Std: ±{score_arr.std() * 100:.2f}%)\n")

print("\nGenerazione Grafici per DECISION TREE")
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), modelli_nested["Decision Tree"][1], cv=cv_inner)
dt_grid.fit(X_arricchito, y)
miglior_dt = dt_grid.best_estimator_

#feture importance Decision Tree
importances_dt = miglior_dt.feature_importances_
nomi_feature = X_arricchito.columns
indices_dt = np.argsort(importances_dt)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importance - Decision Tree", fontsize=14, fontweight='bold')
plt.bar(range(X_arricchito.shape[1]), importances_dt[indices_dt], align="center", color="#3498db")
plt.xticks(range(X_arricchito.shape[1]), [nomi_feature[i] for i in indices_dt], rotation=45, ha='right', fontsize=11)
plt.ylabel('Importanza Relativa', fontsize=12)
plt.tight_layout()
plt.savefig('1_feature_importance_dt.png')
plt.show(block=False)
plt.pause(2)
plt.close()

#Learning Curve Decision Tree
print("Calcolo Curva di Apprendimento DT")
train_sizes_dt, train_scores_dt, test_scores_dt = learning_curve(
    miglior_dt, X_arricchito, y, cv=cv_outer, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')
train_mean_dt = np.mean(train_scores_dt, axis=1)
train_std_dt = np.std(train_scores_dt, axis=1)
test_mean_dt = np.mean(test_scores_dt, axis=1)
test_std_dt = np.std(test_scores_dt, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_dt, train_mean_dt, color='red', marker='o', label='Training score')
plt.fill_between(train_sizes_dt, train_mean_dt - train_std_dt, train_mean_dt + train_std_dt, alpha=0.15, color='red')
plt.plot(train_sizes_dt, test_mean_dt, color='green', marker='o', label='Cross-validation score (Test)')
plt.fill_between(train_sizes_dt, test_mean_dt - test_std_dt, test_mean_dt + test_std_dt, alpha=0.15, color='green')
plt.title('Learning Curve - Decision Tree', fontsize=14, fontweight='bold')
plt.xlabel('Numero di esempi di training', fontsize=12)
plt.ylabel('Accuratezza (Score)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='-', alpha=0.7)
plt.tight_layout()
plt.savefig('2_learning_curve_dt.png')
plt.show(block=False)
plt.pause(2)
plt.close()

print("\nGenerazione Grafici per RANDOM FOREST")
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), modelli_nested["Random Forest"][1], cv=cv_inner)
rf_grid.fit(X_arricchito, y)
miglior_rf = rf_grid.best_estimator_

#Feature importance Random Forest
importances_rf = miglior_rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importance - Random Forest", fontsize=14, fontweight='bold')
plt.bar(range(X_arricchito.shape[1]), importances_rf[indices_rf], align="center", color="#2ecc71")
plt.xticks(range(X_arricchito.shape[1]), [nomi_feature[i] for i in indices_rf], rotation=45, ha='right', fontsize=11)
plt.ylabel('Importanza Relativa', fontsize=12)
plt.tight_layout()
plt.savefig('3_feature_importance_rf.png')
plt.show(block=False)
plt.pause(2)
plt.close()

#Learning curve Random Forest
print("Calcolo Curva di Apprendimento RF")
train_sizes_rf, train_scores_rf, test_scores_rf = learning_curve(
    miglior_rf, X_arricchito, y, cv=cv_outer, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')
train_mean_rf = np.mean(train_scores_rf, axis=1)
train_std_rf = np.std(train_scores_rf, axis=1)
test_mean_rf = np.mean(test_scores_rf, axis=1)
test_std_rf = np.std(test_scores_rf, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_rf, train_mean_rf, color='red', marker='o', label='Training score')
plt.fill_between(train_sizes_rf, train_mean_rf - train_std_rf, train_mean_rf + train_std_rf, alpha=0.15, color='red')
plt.plot(train_sizes_rf, test_mean_rf, color='green', marker='o', label='Cross-validation score (Test)')
plt.fill_between(train_sizes_rf, test_mean_rf - test_std_rf, test_mean_rf + test_std_rf, alpha=0.15, color='green')
plt.title('Learning Curve - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Numero di esempi di training', fontsize=12)
plt.ylabel('Accuratezza (Score)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='-', alpha=0.7)
plt.tight_layout()
plt.savefig('4_learning_curve_rf.png')
plt.show(block=False)
plt.pause(2)
plt.close()

print("\nGenerazione Grafici dei Coefficienti per i Modelli Lineari")

#Coefficineti Logistic Regression
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), modelli_nested["Logistic Regression"][1], cv=cv_inner)
lr_grid.fit(X_arricchito, y)
miglior_lr = lr_grid.best_estimator_

coefficienti_lr = miglior_lr.coef_[0]
indices_lr = np.argsort(np.abs(coefficienti_lr))[::-1]

plt.figure(figsize=(10, 6))
plt.title("Impatto delle Feature - Logistic Regression (Coefficienti)", fontsize=14, fontweight='bold')
plt.bar(range(X_arricchito.shape[1]), coefficienti_lr[indices_lr], align="center", color="#e67e22")
plt.xticks(range(X_arricchito.shape[1]), [nomi_feature[i] for i in indices_lr], rotation=45, ha='right', fontsize=11)
plt.axhline(0, color='black', linewidth=1)
plt.ylabel('Valore del Coefficiente (Peso)', fontsize=12)
plt.tight_layout()
plt.savefig('5_coefficienti_lr.png')
plt.show(block=False)
plt.pause(2)
plt.close()


# coefficienti SVM lineare
svm_grid = GridSearchCV(SVC(kernel='linear', probability=True, random_state=42), modelli_nested["Support Vector Machine"][1], cv=cv_inner)
svm_grid.fit(X_arricchito, y)
miglior_svm = svm_grid.best_estimator_

coefficienti_svm = miglior_svm.coef_[0]
indices_svm = np.argsort(np.abs(coefficienti_svm))[::-1]

plt.figure(figsize=(10, 6))
plt.title("Impatto delle Feature - SVM Lineare (Coefficienti)", fontsize=14, fontweight='bold')
plt.bar(range(X_arricchito.shape[1]), coefficienti_svm[indices_svm], align="center", color="#1abc9c")
plt.xticks(range(X_arricchito.shape[1]), [nomi_feature[i] for i in indices_svm], rotation=45, ha='right', fontsize=11)
plt.axhline(0, color='black', linewidth=1)
plt.ylabel('Valore del Coefficiente (Peso)', fontsize=12)
plt.tight_layout()
plt.savefig('6_coefficienti_svm.png')
plt.show(block=False)
plt.pause(2)
plt.close()

#Learning Curve Logistic Regression
print("Calcolo Curva di Apprendimento LR")
train_sizes_lr, train_scores_lr, test_scores_lr = learning_curve(
    miglior_lr, X_arricchito, y, cv=cv_outer, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')
train_mean_lr = np.mean(train_scores_lr, axis=1)
train_std_lr = np.std(train_scores_lr, axis=1)
test_mean_lr = np.mean(test_scores_lr, axis=1)
test_std_lr = np.std(test_scores_lr, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_lr, train_mean_lr, color='red', marker='o', label='Training score')
plt.fill_between(train_sizes_lr, train_mean_lr - train_std_lr, train_mean_lr + train_std_lr, alpha=0.15, color='red')
plt.plot(train_sizes_lr, test_mean_lr, color='green', marker='o', label='Cross-validation score (Test)')
plt.fill_between(train_sizes_lr, test_mean_lr - test_std_lr, test_mean_lr + test_std_lr, alpha=0.15, color='green')
plt.title('Learning Curve - Logistic Regression', fontsize=14, fontweight='bold')
plt.xlabel('Numero di esempi di training', fontsize=12)
plt.ylabel('Accuratezza (Score)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='-', alpha=0.7)
plt.tight_layout()
plt.savefig('7_learning_curve_lr.png')
plt.show(block=False)
plt.pause(2)
plt.close()

#Learning Curve SVM Lineare
print("Calcolo Curva di Apprendimento SVM")
train_sizes_svm, train_scores_svm, test_scores_svm = learning_curve(
    miglior_svm, X_arricchito, y, cv=cv_outer, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')
train_mean_svm = np.mean(train_scores_svm, axis=1)
train_std_svm = np.std(train_scores_svm, axis=1)
test_mean_svm = np.mean(test_scores_svm, axis=1)
test_std_svm = np.std(test_scores_svm, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_svm, train_mean_svm, color='red', marker='o', label='Training score')
plt.fill_between(train_sizes_svm, train_mean_svm - train_std_svm, train_mean_svm + train_std_svm, alpha=0.15, color='red')
plt.plot(train_sizes_svm, test_mean_svm, color='green', marker='o', label='Cross-validation score (Test)')
plt.fill_between(train_sizes_svm, test_mean_svm - test_std_svm, test_mean_svm + test_std_svm, alpha=0.15, color='green')
plt.title('Learning Curve - SVM Lineare', fontsize=14, fontweight='bold')
plt.xlabel('Numero di esempi di training', fontsize=12)
plt.ylabel('Accuratezza (Score)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='-', alpha=0.7)
plt.tight_layout()
plt.savefig('8_learning_curve_svm.png')
plt.show(block=False)
plt.pause(2)
plt.close()

#Metriche classification report per i modelli
print("\nEstrazione Metriche per i modelli\n")

for nome, y_pred in predizioni_modelli.items():
    print(f"REPORT METRICHE {nome.upper()} (Modello Arricchito)")
    print(classification_report(y, y_pred, target_names=['Commestibile (0)', 'Velenoso (1)']))
    print(f"Parametri migliori: {parametri_migliori[nome]}")
    print("\n")
