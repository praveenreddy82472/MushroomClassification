from mushroom.entity.artifact_entity import ClassificationMetricArtifact
from mushroom.exception import MushroomException
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import os, sys

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        # Convert y_true and y_pred to integers
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred).astype(int)

        # Ensure both y_true and y_pred have only binary values (0, 1)
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        # if you want konw the type of y_true and y_pred
        #print("Unique values in y_true:", unique_true)
        #print("Unique values in y_pred:", unique_pred)

        # Check for any unknown labels
        if not (set(unique_true) <= {0, 1} and set(unique_pred) <= {0, 1}):
            raise ValueError("y_true and y_pred must contain only binary values (0 and 1).")

        # Calculate metrics
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)

        # Return metrics as ClassificationMetricArtifact
        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classification_metric

    except ValueError as ve:
        print("ValueError:", ve)
        raise MushroomException(ve, sys)
    except Exception as e:
        raise MushroomException(e, sys)
