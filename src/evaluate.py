from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        print(f"\n{name} Evaluation:")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Plot and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"outputs/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close()
