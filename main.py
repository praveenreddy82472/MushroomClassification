from mushroom.configuration.mongo_db_connection import MongoDBClient
from mushroom.data_access.mushroom_data import MushroomData
from mushroom.pipeline.training_pipeline import TrainPipeline
from mushroom.constant.database import DATABASE_NAME

if __name__ == '__main__':
    res = TrainPipeline()
    res.run_pipeline()

