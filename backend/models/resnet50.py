import os
import numpy as np
import shutil
import tensorflow as  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
train_data_dir = './datasets/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images'
validation_data_dir = './datasets/ICPR-2022-Small/val'
test_data_dir = './datasets/ICPR-2022-Small/Test'
feedback_dir = 'feedback_images/'
model_save_path = 'resnet50_model.keras'

# Image parameters
img_height, img_width = 224, 224
batch_size = 32
initial_epochs = 10
fine_tune_epochs = 5

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Testing data generator
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the ResNet50 model with pre-trained ImageNet weights, excluding the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Add new top layers for our specific classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Compute class weights to handle class imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
  
# Define the callback to save the model
checkpoint = ModelCheckpoint(
    'resnet50_model_epoch_{epoch:02d}.keras',  # Filename template
    monitor='val_loss',  # Monitor validation loss for saving
    save_best_only=True,  # Save after every epoch
    save_weights_only=False,  # Save the entire model
    mode='auto',
    verbose=1  # Print saving information
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=initial_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    class_weight=class_weights,
    callbacks=[checkpoint]  # Include the callback
)

# Save the final model
model.save('resnet50_model_final.keras')
# Evaluate the model
val_labels = validation_generator.classes
val_predictions = model.predict(validation_generator)
val_pred_labels = np.argmax(val_predictions, axis=1)

print("Initial Classification Report:")
print(classification_report(val_labels, val_pred_labels, target_names=list(train_generator.class_indices.keys())))

# Feedback Loop
def feedback_loop(model, validation_generator, feedback_dir):
    val_labels = validation_generator.classes
    val_predictions = model.predict(validation_generator)
    val_pred_labels = np.argmax(val_predictions, axis=1)
    
    if not os.path.exists(feedback_dir):
        os.makedirs(feedback_dir)
    
    misclassified_count = 0
    for i, (pred, true) in enumerate(zip(val_pred_labels, val_labels)):
        if pred != true:
            misclassified_count += 1
            img_path = validation_generator.filepaths[i]
            class_name = list(validation_generator.class_indices.keys())[true]
            dest_dir = os.path.join(feedback_dir, class_name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            shutil.copy(img_path, dest_dir)
    
    print(f"Misclassified {misclassified_count} images. Move them to the correct class folders in {feedback_dir} for retraining.")
    return misclassified_count

# Call the feedback loop function
misclassified_count = feedback_loop(model, validation_generator, feedback_dir)

# Retrain with feedback if there are misclassified images
if misclassified_count > 0:
    
    
    # Reload the data generators to include feedback images
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Fine-tune the model with a lower learning rate
    for layer in base_model.layers[-10:]:  # Unfreeze the last 10 layers
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=validation_generator
    )

# Save the updated model
model.save(model_save_path)
print(f"Updated model saved to {model_save_path}")

# Test the model
test_labels = test_generator.classes
test_predictions = model.predict(test_generator)
test_pred_labels = np.argmax(test_predictions, axis=1)

print("Test Classification Report:")
print(classification_report(test_labels, test_pred_labels, target_names=list(train_generator.class_indices.keys())))