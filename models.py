from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Dropout
from keras.models import model_from_json


def save_model(model, name='model'):
    """
    Sauvegarde le modèle dans model.json et les poids dans model.h5
    :param model: modèle Keras
    :param name: nom des fichiers de sauvegarde
    """
    model_json = model.to_json()
    with open(name + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + '.h5')
    print("Saved model to disk")
 

def load_model(name='model'):
    """
    Charge le modèle et les poids associés
    :param name: nom des fichiers de sauvegarde
    :return: loaded_model
    """
    # load json and create model
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + '.h5')
    print("Loaded model from disk")
    return loaded_model


def CNN(X0, Y0, X1, Y1, fit=False):
    """
    Modèle CNN et Dense
    Les paramètres d'entrées sont les sortie de preprocessing.preprocess
    :return: model
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    if fit:
        model.fit(X0, Y0, epochs=10, batch_size=128)
        print(model.evaluate(X1, Y1))

    return model
