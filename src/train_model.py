import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Paths
csv_file = 'data/trainLabels.csv'
base_img_dir = 'data/sampleimages/'
output_dir = 'data/preprocessed_images'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load CSV and Create Full Image Paths
data = pd.read_csv(csv_file)  # Assumes columns 'image_name' and 'rating'
data['full_path'] = data['image_name'].apply(lambda x: os.path.join(base_img_dir, x))

# Step 2: Preprocess and Organize Images into Label-specific Folders
def preprocess_and_save_images():
    for _, row in tqdm(data.iterrows(), total=len(data)):
        img_path = row['full_path']
        label = row['rating']  # DR level

        # Load and preprocess the image
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (299, 299))  # Resize to model input size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize

        # Save the preprocessed image in the corresponding label directory
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        cv2.imwrite(os.path.join(label_dir, os.path.basename(img_path)), img * 255)

print("Starting image preprocessing...")
preprocess_and_save_images()
print("Image preprocessing complete.")

# Step 3: Setup ImageDataGenerator for Training and Validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    output_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    output_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Step 4: Define the Model
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)  # 5 classes for DR levels

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the Model
print("Starting model training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('models/best_inceptionv3_model.h5', save_best_only=True, monitor='val_loss', mode='min'),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
    ]
)
print("Model training complete.")

# Save the final model
model.save('models/inceptionv3_model.h5')
print("Model saved.")
