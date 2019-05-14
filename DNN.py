import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np


def dnn(train_x, train_y,  valid_x, valid_y, test_x, test_y, input_dim, num_layers=1, batch_size=256, max_epochs=300):
    model = Sequential()
    for i in range(num_layers):
        model.add(Dense(256, activation='relu', input_dim=input_dim))#kernel_initializer='uniform',
    model.add(Dense(61, activation='sigmoid'))
    print(model.summary())
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    early_stopping = [EarlyStopping(monitor='val_loss', patience=10, verbose=2)]
    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        verbose=1,
                        validation_data=(valid_x, valid_y),
                        callbacks=early_stopping)
    # evaluate
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    '''
    code for download from google drive and run it in colab:
    ------
    from google.colab import drive
    drive.mount('/content/gdrive')
    ------
    file_name = "/content/gdrive/My Drive/data/"
    '''
    # ----- load preprocessed training data and targets
    file_name = ''
    train_x_lmfcc = np.load(file_name + 'orignal_lmfcc_train_X.dat', allow_pickle=True)
    # train_x_mspec = np.load(file_name + 'original_mspec_train_X.dat', allow_pickle=True)
    train_y = np.load(file_name + 'train_y.dat', allow_pickle=True)[:1353828, :]

    valid_x_lmfcc = np.load(file_name + 'orignal_lmfcc_val_X.dat', allow_pickle=True)
    # valid_x_mspec = np.load(file_name + 'original_mspec_val_X.dat', allow_pickle=True)
    valid_y = np.load(file_name + 'val_y.dat', allow_pickle=True)[:149381, :]

    test_x_lmfcc = np.load(file_name + 'orignal_lmfcc_test_X.dat', allow_pickle=True)
    # test_x_mspec = np.load(file_name + 'original_mspec_test_X.dat', allow_pickle=True)
    test_y = np.load(file_name + 'test_y.dat', allow_pickle=True)[:1522906, :]

    # print(train_x_lmfcc.shape, train_y.shape, valid_x_lmfcc.shape, valid_y.shape, test_x_lmfcc.shape, test_y.shape)
    # (1353828, 13) (1353828, 61) (149381, 13) (149381, 61) (1522906, 13) (1522906, 61)

    # ------ try 1,(2,3,) 4 layers for the following input ------
    # input: liftered MFCCs
    dnn(num_layers=1, train_x=train_x_lmfcc, train_y=train_y, valid_x=valid_x_lmfcc, valid_y=valid_y,
        test_x=test_x_lmfcc, test_y=test_y, input_dim=13, max_epochs=10)
    dnn(num_layers=4, train_x=train_x_lmfcc, train_y=train_y, valid_x=valid_x_lmfcc, valid_y=valid_y,
        test_x=test_x_lmfcc, test_y=test_y, input_dim=13, max_epochs=10)

    # input: filterbank features
    # dnn(num_layers=1, train_x=train_x_mspec, train_y=train_y, valid_x=valid_x_mspec, valid_y=valid_y, test_x=test_x_mspec, test_y=test_y, input_dim=40)
    # dnn(num_layers=4, train_x=train_x_mspec, train_y=train_y, valid_x=valid_x_mspec, valid_y=valid_y, test_x=test_x_mspec, test_y=test_y, input_dim=40)

    train_x_lmfcc_d = np.load(file_name + 'lmfcc_train_X.dat', allow_pickle=True)  # (1353828, 91)
    # train_x_mspec_d = np.load(file_name + 'mspec_train_x.dat', allow_pickle=True)
    train_y = np.load(file_name + 'train_y.dat', allow_pickle=True)  # (1373520, 61)

    valid_x_lmfcc_d = np.load(file_name + 'lmfcc_val_X.dat', allow_pickle=True)  # (149381, 91)
    # valid_x_mspec_d = np.load(file_name + 'mspec_val_x.dat', allow_pickle=True)
    valid_y = np.load(file_name + 'val_y.dat', allow_pickle=True)  # (152751, 61)

    test_x_lmfcc_d = np.load(file_name + 'lmfcc_test_X.dat', allow_pickle=True)  # (1522906, 91)
    # test_x_mspec_d = np.load(file_name + 'mspec_test_x.dat', allow_pickle=True)
    test_y = np.load(file_name + 'test_y.dat', allow_pickle=True)  # (1539900, 61)

    # ------ try 1,(2,3,) 4 layers for the following input ------
    # input: dynamic lmfcc features
    dnn(num_layers=1, train_x=train_x_lmfcc_d, train_y=train_y, valid_x=valid_x_lmfcc_d, valid_y=valid_y,
        test_x=test_x_lmfcc_d, test_y=test_y, input_dim=91, max_epochs=10)
    dnn(num_layers=4, train_x=train_x_lmfcc_d, train_y=train_y, valid_x=valid_x_lmfcc_d, valid_y=valid_y,
        test_x=test_x_lmfcc_d, test_y=test_y, input_dim=91, max_epochs=10)

    # input: dynamic mspec features
    # dnn(num_layers=1, train_x=train_x_mspec_d, train_y=train_y, valid_x=valid_x_mspec_d, valid_y=valid_y, test_x=test_x_mspec_d, test_y=test_y, input_dim=280)
    # dnn(num_layers=4, train_x=train_x_mspec_d, train_y=train_y, valid_x=valid_x_mspec_d, valid_y=valid_y, test_x=test_x_mspec_d, test_y=test_y, input_dim=280)

    # TODO: 5.1 Detailed Evaluation

