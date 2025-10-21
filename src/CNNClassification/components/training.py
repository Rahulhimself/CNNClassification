from src.CNNClassification.entity import TrainingConfig
import tensorflow as tf
from pathlib import Path


import tensorflow as tf
from pathlib import Path

# NOTE: This assumes 'TrainingConfig' and necessary imports are available.

class Training:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None
    
    # 1. MODIFICATION: Load Model without old optimizer state (Fixes ValueError)
    def get_base_model(self):
        """Loads the pre-trained model architecture and weights, discarding the old optimizer state."""
        print(f"Loading base model from: {self.config.updated_base_model_path}")
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False  # <--- CRITICAL: Prevents loading the saved optimizer state
        )
    
    # 2. ADDITION: Compile Model with a new optimizer (Fixes 'Must call compile()' error)
    def prepare_full_model(self):
        """Creates a NEW optimizer instance and compiles the model."""
        
        # Ensure learning rate is defined (Fixes NameError)
        try:
            learning_rate = self.config.params_learning_rate 
        except AttributeError:
            # Fallback if config is misnamed
            print("Warning: 'params_learning_rate' not found in config. Using default 0.001.")
            learning_rate = 0.001 
        
        # Instantiate a brand NEW optimizer object (Fixes ValueError)
        new_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
        
        # Compile the model
        self.model.compile(
            optimizer=new_optimizer,  
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
    # 3. train_valid_generator remains the same (using ImageDataGenerator)
    def train_valid_generator(self):
        """Sets up the training and validation data generators."""
        
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )
        # ... (rest of the train_valid_generator code remains the same) ...
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        print("Creating validation generator...")
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator
            
        print("Creating training generator...")
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    # 4. save_model remains the same
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    # 5. train method remains the same
    def train(self, callback_list: list):
        """Fits the model to the training data."""
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )







# Assuming 'training' object is initialized and 'callback_list' is defined

