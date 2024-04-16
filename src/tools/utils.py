import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
from glob import glob
from dotenv import load_dotenv
import psycopg2

load_dotenv()


def plot_confusion_matrix(cm, class_names, normalize=True):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    np.seterr(divide="ignore", invalid="ignore")

    # Normalize the confusion matrix.
    if normalize:
        cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    figure = plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return figure


def plot_pr_curve(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    np.seterr(divide="ignore", invalid="ignore")

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_files(d, pattern, sort=True):
    """
    Return a list of files in a given directory.

    Args:
        d (str): The path to the directory.
        pattern (str): The wildcard to filter files with.
        sort (bool): Whether to sort the returned list.
    """
    files = glob(os.path.join(d, pattern))
    files = [f for f in files if os.path.isfile(f)]

    file_names = [x.split("/")[-1].split(".")[0] for x in files]
    if "best" in file_names:
        del files[file_names.index("best")]

    if sort:
        files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    return files


def get_db_connection():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )

    return conn


def query_database(query):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(query)
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return columns, rows
