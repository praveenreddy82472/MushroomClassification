
from mushroom.utils.main_utils import load_numpy_array_data
from mushroom.exception import MushroomException
from mushroom.logger import logging
from mushroom.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from mushroom.entity.config_entity import ModelTrainerConfig
import os,sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from mushroom.ml.metric.classification_metric import get_classification_score
from mushroom.ml.model.estimator import MushroomModel
from mushroom.utils.main_utils import save_object,load_object
class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
        data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise MushroomException(e,sys)

    def perform_hyper_paramter_tunig(self):...
    

    def train_model(self,x_train,y_train):
        try:
             # Define Logistic Regression model
             log_reg = LogisticRegression(max_iter=1000)
             # Set up hyperparameters to tune
             param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
                    'solver': ['liblinear', 'lbfgs']  # Different solvers for optimization
                }
        
            # Set up GridSearchCV
             grid_search = GridSearchCV(log_reg, param_grid, cv=5, verbose=1, n_jobs=-1)
        
            # Train the model using GridSearchCV
             grid_search.fit(x_train, y_train)
        
            # Return the best estimator
             return grid_search.best_estimator_
        except Exception as e:
            raise e
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model = self.train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)
            classification_train_metric =  get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            if classification_train_metric.f1_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good to provide expected accuracy")
            
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)


            #Overfitting and Underfitting
            diff = abs(classification_train_metric.f1_score-classification_test_metric.f1_score)
            
            if diff>self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            sensor_model = MushroomModel(preprocessor=preprocessor,model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=sensor_model)

            #model trainer artifact

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise MushroomException(e,sys)