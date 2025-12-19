import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
import pandas as pd
import datetime as dt
import json

from tqdm import tqdm
from sklearn.metrics import confusion_matrix 
import keras
from keras import layers, Sequential
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import get_file
from pathlib import Path
import os
FONTDICT = {    
    'fontsize': 14,
    'fontweight': 'bold',
    'color': 'darkblue',
    "fontfamily": "monospace"
}

def tensorflow_example(results_path: Path):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_path / timestamp
    results_path.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
    EPOCHS = int(os.getenv("EPOCHS", 25))
    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) =  keras.datasets.cifar10.load_data()


    # Normalizing pixel values to be btw 0 and 1
    X_TRAIN, X_TEST = X_TRAIN/255.0, X_TEST/255.0    
    classes = (
        'airplane', 
        'automobile', 
        'bird', 
        'cat', 
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    )

    fig, ax = plt.subplots(5,5, figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_TRAIN[i])
        plt.xlabel(classes[Y_TRAIN[i][0]], fontdict=FONTDICT)
    plt.savefig(results_path / "sample_images_tensorflow.png")
    plt.close()


    model: Sequential = Sequential([
        layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.summary()    
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10, 
        restore_best_weights=True, 
        verbose=1
    )

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history: keras.callbacks.History = model.fit(
        X_TRAIN, 
        Y_TRAIN, 
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[
            early_stopping
        ]
    )

    pltdf = pd.DataFrame(history.history)
    plt.figure(figsize=(10,6))
    plt.title("Training and Validation Accuracy", fontdict=FONTDICT)    
    plt.plot(pltdf.index, pltdf["accuracy"], label="Training Accuracy")
    plt.plot(pltdf.index, pltdf["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs", fontdict=FONTDICT)
    plt.ylabel("Accuracy", fontdict=FONTDICT)
    plt.grid(True)
    plt.legend()
    plt.savefig(results_path / "accuracy_plot.svg", format="svg")
    plt.close()

    test_loss, test_acc = model.evaluate(X_TEST, Y_TEST, verbose=2)
    print(f"Test accuracy: {test_acc:.3f}")

    # confusion matrix
    y_pred = model.predict(X_TEST)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(Y_TEST, y_pred_classes)
    plt.figure(figsize=(10,8))
    plt.title("Confusion Matrix", fontdict=FONTDICT)
    plt.ylabel('Actual', fontdict=FONTDICT)
    plt.xlabel('Predicted', fontdict=FONTDICT)
    sns.heatmap(
        cm, 
        annot=True, 
        annot_kws={
            "fontsize":12,
            "fontweight":"bold",
            "fontfamily":"monospace"
        },
        fmt='d', 
        cmap='Blues',
        cbar=True,
        linewidths=0.5,
        linecolor='black',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.savefig(results_path / "confusion_matrix.png")
    plt.close()

    with open(results_path / "tensorflow_results.json", "w") as f:
        json.dump({
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "n_test_samples": len(Y_TEST),
            "n_correct": int(test_acc * len(Y_TEST)),
            "epochs": EPOCHS,
            "actual_epochs": len(history.history['loss']),
            "batch_size": BATCH_SIZE
        }, f, indent=4)

    # save the model
    model.save(results_path / "cnn_cifar10_tensorflow.keras")

if __name__ == "__main__":
    tensorflow_example(results_path=Path("./results"))