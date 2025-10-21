from src.CNNClassification.config import ConfigurationManager
from src.CNNClassification.components import Training
from src.CNNClassification import logger
from CNNClassification.components.prepare_callback import PrepareCallback

class ModelTrainingPipeline:
    def main(self):
        config_manager = ConfigurationManager()
        training_config = config_manager.get_training_config()
        prepare_callbacks_config = config_manager.get_prepare_callback_config() # <-- Get the config

        # 1. CREATE the Training component instance
        training_component = Training(config=training_config) 
        
        # 2. CREATE the PrepareCallback component instance
        callback_component = PrepareCallback(config=prepare_callbacks_config)
        
        # 3. CRITICAL: CALL THE METHOD to get the list of Keras Callbacks
        callback_list = callback_component.get_tb_ckpt_callbacks() 
        
        # 4. Run the rest of the pipeline
        training_component.get_base_model()
        training_component.prepare_full_model()
        training_component.train_valid_generator()
        
        # 5. Pass the actual LIST of callbacks to train
        training_component.train(callback_list=callback_list)
