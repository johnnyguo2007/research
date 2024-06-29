import tensorflow as tf
import time
#  test header 

def train_model(device):
    with tf.device(device):
        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        # Build a simple convolutional network
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Train the model
        start_time = time.time()
        model.fit(x_train, y_train, epochs=5, validation_split=0.1)
        end_time = time.time()

        return end_time - start_time


# Measure time on GPU
gpu_time = train_model('/gpu:0')
print("Training time on GPU:", gpu_time)

# Measure time on CPU
cpu_time = train_model('/cpu:0')
print("Training time on CPU:", cpu_time)
