from src.cnnClassifier import logger
from src.cnnClassifier.utils.comman import get_size
from src.cnnClassifier.entity.config import DataIngestionConfig
from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.data_ingestion_01 import DataIngestion
from src.cnnClassifier.pipeline.data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e