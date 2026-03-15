import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from datasets.data_pipline import load_datasets

# Load datasets

train_ds, val_ds, test_ds = load_datasets(batch_size=16, img_size=(224,224))


# Build model

num_classes = 17


base_model = EfficientNetB0(
    include_top=False,
    input_shape=(224,224,3),
    weights='imagenet'
)
base_model.trainable = False


inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)


# Compile model

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Train model (initial training)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8
)


# Fine-tuning: unfreeze some layers
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8
)


# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc*100:.2f}%")


# Save the trained model
#model.save("plant_classifier_model")