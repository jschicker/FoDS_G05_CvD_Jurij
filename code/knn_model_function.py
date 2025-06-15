from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve

data = pd.read_csv(
        filepath_or_buffer="../data/2025_cardio_train.csv",
        index_col=0,
        na_filter=False,
        dtype = {"gender": "category",
                 "cholesterol": "category",
                 "gluc": "category",
                 "smoke": "category",
                 "alco": "category",
                 "active": "category",
                 "cardio": "category",
        }
)
data["cholesterol"] = data["cholesterol"].cat.as_ordered()
data["gluc"] = data["gluc"].cat.as_ordered()

data_clone = data.copy()
data = data[data["ap_hi"] <= 200 ]
data = data[data["ap_hi"] >= 0]
data = data[data["ap_lo"] <= 200]
data = data[data["ap_lo"] >= 0]


data_encoded = pd.get_dummies(data, drop_first=True, )

X = data_encoded.drop("cardio_1", axis=1)
y = data_encoded["cardio_1"]

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



def knn_predictions(X_train,X_test,y_train):

    # Create a pipeline: scaling + KNN
    pipe = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the features
    ('knn', KNeighborsClassifier())  # Step 2: Apply KNN
    ])

    # Set optimized parameters
    knn_metric = "manhattan"
    n_k_neighbours = 29
    knn_weights = "uniform"

    pipe.set_params(knn__n_neighbors=int(n_k_neighbours),
                knn__weights=knn_weights,
                knn__metric=knn_metric)

    # fit and return predictions
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    return y_pred, y_proba

predictions, probability = knn_predictions(X_train,X_test,y_train)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

auc = roc_auc_score(y_test, probability)
print("AUC:", auc)

fpr, tpr, thresholds = roc_curve(y_test, probability)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()