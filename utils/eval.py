from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_metrics_report(y_test, y_pred, labels):
    report = classification_report(y_test, y_pred, target_names=labels)
    print(report)
    return report

def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


def plot_wrong_predictions(model, test_ds, labels):

    image_batch, label_batch = next(iter(test_ds))
    y_pred_probs = model.predict(image_batch)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # find wrong predictions
    arr = (label_batch.numpy().flatten() == y_pred)
    indices = np.where(arr == False)[0]

    n_images = min(len(indices), 10)
    cols = 5
    rows = n_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))


    for i, ax in enumerate(axes.flat):
        if i >= n_images:
            break

        index = indices[i]
        image = image_batch[index].numpy()/255
        label = labels[label_batch[index].numpy().flatten()[0]]


        ax.imshow(image)
        true_name = label
        pred_name = labels[y_pred[index]]
        correct = true_name == pred_name

        ax.set_title(
            f"True: {true_name}\nPred: {pred_name}",
            color="green" if correct else "red",
            fontsize=9
        )
        ax.axis("off")

    plt.tight_layout()
    # plt.savefig("predictions.png")
    plt.show()


def plot_predictions(model, test_ds, labels):

    n_labels = len(labels)
    cols = 5
    rows = n_labels // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))

    image_batch, label_batch = next(iter(test_ds))
    y_pred_probs = model.predict(image_batch)
    y_pred = np.argmax(y_pred_probs, axis=1)


    for i, ax in enumerate(axes.flat):
        if i >= n_labels:
            break

        image = image_batch[i].numpy()/255
        label = labels[label_batch[i].numpy().flatten()[0]]


        ax.imshow(image)
        true_name = label
        pred_name = labels[y_pred[i]]
        correct = true_name == pred_name

        ax.set_title(
            f"True: {true_name}\nPred: {pred_name}",
            color="green" if correct else "red",
            fontsize=9
        )
        ax.axis("off")

    plt.tight_layout()
    # plt.savefig("predictions.png")
    plt.show()

