import pandas as pd
import numpy as np
import joblib
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report #

def plot_confusion_matrix(y_true, y_pred, class_labels, save_path):
    """Gera e salva a matriz de confusão normalizada."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, cmap="viridis", fmt=".2f",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Normalized Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="Caminho para o features.csv")
    args = parser.parse_args()

    # Carregamento de dados
    dataset = pd.read_csv(args.file_path)
    X = dataset.drop(columns=['label', 'file_name'])
    y = dataset['label']
    
    # Validação Cruzada Estratificada
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_pred = cross_val_predict(svm_model, X, y, cv=cv)
    
    # Métricas
    print(f"Acurácia Geral: {accuracy_score(y, y_pred):.4f}")
    print(classification_report(y, y_pred))
    
    # Salvar Modelo Final
    os.makedirs("results", exist_ok=True)
    svm_model.fit(X, y)
    joblib.dump(svm_model, "results/svm_final.pkl")
    
    plot_confusion_matrix(y, y_pred, np.unique(y), "results/cm_final.png")
    print("Modelo e Matriz de Confusão salvos na pasta /results")

if __name__ == "__main__":
    main()
