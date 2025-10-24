import tensorflow as tf
from pathlib import Path
from src.CNNClassification.entity import EvaluationConfig
from src.CNNClassification.utils import save_json



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = tf.keras.models.load_model(self.config.path_of_model) 
        print(f"Model loaded successfully from: {self.config.path_of_model}")
    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            #Using 'sparse' to generate integer labels (shape (None, 1))
            class_mode='sparse',
            **dataflow_kwargs
        )

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
            self._valid_generator() # 1. Prepare the generator

            # 2. Re-compile the model with the correct loss function
            # This step is CRUCIAL to match the 'sparse' label generation
            self.model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"] 
                # You can also use: loss='sparse_categorical_crossentropy'
            )
            
            # 3. Perform evaluation (This is where the original traceback error occurred)
            self.score = self.model.evaluate(self.valid_generator)

            # 4. Save results (Placeholder for your original code's result saving logic)
            # self.save_score() # Assuming you have a method to save the results
            print(f"Evaluation Loss: {self.score[0]}, Accuracy: {self.score[1]}")

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)