import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline


from mushroom.constant.training_pipeline import TARGET_COLUMN
from mushroom.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
    
)
from mushroom.entity.config_entity import DataTransformationConfig
from mushroom.exception import MushroomException
from mushroom.logger import logging
from mushroom.ml.model.estimator import TargetValueMapping
from mushroom.utils.main_utils import save_numpy_array_data, save_object


pd.set_option('future.no_silent_downcasting', True)



from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        """
        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise MushroomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MushroomException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            # OneHotEncoder for categorical variables
            onehot_encoder = OneHotEncoder()
            scaler = StandardScaler(with_mean=False)
            pca = PCA(n_components=10)  # Adjust n_components as needed

            # Define the column transformer with one-hot encoding followed by PCA
            preprocessor = Pipeline(
                steps=[
                    ("OneHotEncoding", onehot_encoder),
                    ("StandardScaler",scaler),
                    ("PCA", pca)
                ]
            )
            return preprocessor

        except Exception as e:
            raise MushroomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            preprocessor = self.get_data_transformer_object()

            # Splitting input features and target labels
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())

            # Applying preprocessor (OneHotEncoding and PCA)
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            
            # Convert target variable to integer
            target_feature_train_df = target_feature_train_df.astype(int)
            # Convert target variable to integer
            target_feature_test_df = target_feature_test_df.astype(int) 
            
            print(transformed_input_train_feature.dtype)
            print(target_feature_train_df.dtype)
            print(transformed_input_test_feature.dtype)
            print(target_feature_test_df.dtype)
            


            # Apply SMOTETomek on transformed data
            smt = SMOTETomek(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )
            print(target_feature_train_final.dtypes)  # Should be int or object
            print(target_feature_train_final.unique())  # Check if values are discrete, e.g., [0, 1]

            # Save the transformed data arrays
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            # Preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise MushroomException(e, sys) from e
