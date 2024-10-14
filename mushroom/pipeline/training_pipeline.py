from mushroom.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig

from mushroom.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact

from mushroom.exception import MushroomException
import sys,os
from mushroom.logger import logging
from mushroom.components.data_ingestion import DataIngestion
from mushroom.components.data_validation import DataValidation


class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        #self.s3_sync = S3Sync()
        


    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except  Exception as e:
            raise  MushroomException(e,sys)
        
    def start_data_validaton(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config = data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except  Exception as e:
            raise  MushroomException(e,sys)
    
    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running=True

            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact=self.start_data_validaton(data_ingestion_artifact=data_ingestion_artifact)
        except  Exception as e:
            self.sync_artifact_dir_to_s3()
            TrainPipeline.is_pipeline_running=False
            raise  MushroomException(e,sys)