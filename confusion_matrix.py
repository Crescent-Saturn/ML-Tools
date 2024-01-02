import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# compute from scratch
def confusion_matrix(y_true, y_pred):

    true = np.ravel(y_true)
    pred = np.ravel(y_pred)

    k = len(np.unique(true))

    cm = np.zeros((k, k))

    for i in range(len(true)):
        cm[true[i], pred[i]] += 1
    
    return cm.astype(int)

y_true = [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0]]
y_pred = [[0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1]]


CM = confusion_matrix(y_true, y_pred)

# compute precision, recall, etc from scratch
TP = CM.diagonal()
FP = CM.sum(axis=0) - TP
FN = CM.sum(axis=1) - TP

precision = TP / (TP+FP)
recall = TP / (TP+FN)

f1_score = 2*precision*recall / (precision+recall)
iou = TP / (TP+FN+FP)

print(f"precision: {precision}\n recall: {recall}\n F1-Score: {f1_score}\n IoU: {iou}")

# Get reference from sklearn
print(classification_report(y_true, y_pred))

ConfusionMatrixDisplay.from_predictions(np.ravel(y_true), np.ravel(y_pred), cmap='Blues')