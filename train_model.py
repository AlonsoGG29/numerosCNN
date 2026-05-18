import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np

print("Cargando datos MNIST...")
(x_train_full, y_train_full), (x_test, y_test) = datasets.mnist.load_data()

print("Preprocesando datos...")
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Redimensionamiento: Añadir canal de color (n, 28, 28, 1)
x_train_full = np.expand_dims(x_train_full, -1)
x_test = np.expand_dims(x_test, -1)

# División de datos (51000 para entrenamiento, 9000 para validación, igual que en el notebook)
split_idx = 51000
x_train, x_val = x_train_full[:split_idx], x_train_full[split_idx:]
y_train, y_val = y_train_full[:split_idx], y_train_full[split_idx:]

print(f"Set de Entrenamiento: {x_train.shape}")
print(f"Set de Validación: {x_val.shape}")

print("Definiendo el modelo CNN...")
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model_improved = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model_improved.compile(optimizer="adam", 
                      loss="sparse_categorical_crossentropy", 
                      metrics=["accuracy"])

print("Entrenando el modelo (5 épocas para rapidez)...")
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model_improved.fit(
    x_train, y_train, 
    epochs=5, 
    batch_size=64, 
    validation_data=(x_val, y_val),
    callbacks=[callback]
)

print("Guardando el modelo...")
model_improved.save("modelo_digitos_postal.keras")
print("Modelo guardado exitosamente como modelo_digitos_postal.keras.")
