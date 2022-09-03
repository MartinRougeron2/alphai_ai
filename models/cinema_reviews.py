import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from keras import layers
from keras import losses

from keras.utils.vis_utils import plot_model


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


class TextClassification:
    def __init__(self):
        batch_size = 32
        seed = 42

        raw_train_ds = tf.keras.utils.text_dataset_from_directory(
            "aclImdb/train",
            batch_size=batch_size,
            validation_split=0.2,
            subset="training",
            seed=seed,
        )

        raw_val_ds = tf.keras.utils.text_dataset_from_directory(
            "aclImdb/train",
            batch_size=batch_size,
            validation_split=0.2,
            subset="validation",
            seed=seed,
        )

        raw_test_ds = tf.keras.utils.text_dataset_from_directory(
            "aclImdb/test", batch_size=batch_size
        )

        max_features = 10000
        sequence_length = 250

        vectorize_layer = layers.TextVectorization(
            standardize=custom_standardization,
            max_tokens=max_features,
            output_mode="int",
            output_sequence_length=sequence_length,
        )

        train_text = raw_train_ds.map(lambda x, y: x)
        vectorize_layer.adapt(train_text)

        def vectorize_text(text, label):
            text = tf.expand_dims(text, -1)
            return vectorize_layer(text), label

        train_ds = raw_train_ds.map(vectorize_text)
        val_ds = raw_val_ds.map(vectorize_text)
        test_ds = raw_test_ds.map(vectorize_text)

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        embedding_dim = 16

        model = tf.keras.Sequential(
            [
                layers.Embedding(max_features + 1, embedding_dim),
                layers.Dropout(0.2),
                layers.GlobalAveragePooling1D(),
                layers.Dropout(0.2),
                layers.Dense(1),
            ]
        )

        model.compile(
            loss=losses.BinaryCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=tf.metrics.BinaryAccuracy(threshold=0.0),
        )

        plot_model(model, to_file='model_plot_TextClass.png', show_shapes=True, show_layer_names=True)

        model.summary()

        epochs = 10
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        loss, accuracy = model.evaluate(test_ds)

        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

        export_model = tf.keras.Sequential(
            [vectorize_layer, model, layers.Activation("sigmoid")]
        )

        export_model.compile(
            loss=losses.BinaryCrossentropy(from_logits=False),
            optimizer="adam",
            metrics=["accuracy"],
        )

        # Test it with `raw_test_ds`, which yields raw strings
        loss, accuracy = export_model.evaluate(raw_test_ds)
        print(accuracy)

        self.model = export_model

    def test(self, text=None):
        print("predictions\n\n\n\n\n")
        while True:
            predicts = self.model.predict([input()])
            if predicts[0] >= 0.42:
                print("positive at " + str(predicts[0]) + "% positivity")
            else:
                print("negative at " + str(predicts[0]) + "% positivity")


model_1 = TextClassification()
model_1.test()
