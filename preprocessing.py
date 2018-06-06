from keras.datasets import mnist
from keras.utils import np_utils


def load():
    """
    Charge labase MNIST de Keras
    :return: X0, y0, X1, y1
    """
    data = mnist.load_data()
    (X0, y0), (X1, y1) = data
    return X0, y0, X1, y1


def preprocess_X(X):
    """
    Reshape de X pour l'adapter au CNN de Keras et normalisation
    :param X: data de MNIST
    :return: X
    """
    X = X.reshape(X.shape[0], 28, 28, 1).astype('float')
    X /= 255
    return X


def preprocess_y(y):
    """
    Convertie en catégories par colonne les labels de MNIST
    :param y: label de MNIST
    :return: y
    """
    Y = np_utils.to_categorical(y, 10)
    return Y


def preprocess():
    """
    Applique le chargement des données et les fonctions preprocess
    :return: X0, y0, Y0, X1, y1, Y1
    où y0 et y1 sont les labels d'origines (entre 0 et 9)
    alors que Y0 et Y1 sont des catégories (valeur 0 ou 1)
    """

    X0, y0, X1, y1 = load()
    X0 = preprocess_X(X0)
    X1 = preprocess_X(X1)
    Y0 = preprocess_y(y0)
    Y1 = preprocess_y(y1)

    return X0, y0, Y0, X1, y1, Y1