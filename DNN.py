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
    print(model.summary())
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
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
    train_x_lmfcc_d = np.load(file_name + 'lmfcc_train_x.dat', allow_pickle=True)
    train_x_mspec_d = np.load(file_name + 'mspec_train_x.dat', allow_pickle=True)
    train_y = np.load(file_name + 'targets_train_x.dat',allow_pickle=True)
    valid_x_lmfcc_d = np.load(file_name + 'lmfcc_valid_x.dat', allow_pickle=True)
    valid_x_mspec_d = np.load(file_name + 'mspec_valid_x.dat', allow_pickle=True)
    valid_y = np.load(file_name + 'valid_y.dat', allow_pickle=True)
    test_x_lmfcc_d = np.load(file_name + 'lmfcc_test_x.dat', allow_pickle=True)
    test_x_mspec_d = np.load(file_name + 'mspec_test_x.dat', allow_pickle=True)
    test_y = np.load(file_name + 'targets_test_x.dat',allow_pickle=True)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')
    train_x_lmfcc = np.load(file_name + 'originallmfcc_train_x.dat', allow_pickle=True)
    train_x_mspec = np.load(file_name + 'originalmspec_train_x.dat', allow_pickle=True)
    valid_x_lmfcc = np.load(file_name + 'originallmfcc_valid_x.dat', allow_pickle=True)
    valid_x_mspec = np.load(file_name + 'originalmspec_valid_x.dat', allow_pickle=True)
    test_x_lmfcc = np.load(file_name + 'originallmfcc_test_x.dat', allow_pickle=True)
    test_x_mspec = np.load(file_name + 'originalmspec_test_x.dat', allow_pickle=True)

    # ------ try 1,(2,3,) 4 layers for the following input ------
    # input: liftered MFCCs
    dnn(num_layers=1, train_x=train_x_lmfcc, train_y=train_y, valid_x=valid_x_lmfcc, valid_y=valid_y, test_x=test_x_lmfcc, test_y=test_y, input_dim=13)
    dnn(num_layers=4, train_x=train_x_lmfcc, train_y=train_y, valid_x=valid_x_lmfcc, valid_y=valid_y, test_x=test_x_lmfcc, test_y=test_y, input_dim=13)

    # input: filterbank features
    dnn(num_layers=1, train_x=train_x_mspec, train_y=train_y, valid_x=valid_x_mspec, valid_y=valid_y, test_x=test_x_mspec, test_y=test_y, input_dim=40)
    dnn(num_layers=4, train_x=train_x_mspec, train_y=train_y, valid_x=valid_x_mspec, valid_y=valid_y, test_x=test_x_mspec, test_y=test_y, input_dim=40)

    # input: dynamic lmfcc features
    dnn(num_layers=1, train_x=train_x_lmfcc_d, train_y=train_y, valid_x=valid_x_lmfcc_d, valid_y=valid_y, test_x=test_x_lmfcc_d, test_y=test_y, input_dim=91)
    dnn(num_layers=4, train_x=train_x_lmfcc_d, train_y=train_y, valid_x=valid_x_lmfcc_d, valid_y=valid_y, test_x=test_x_lmfcc_d, test_y=test_y, input_dim=91)

    # input: dynamic mspec features
    dnn(num_layers=1, train_x=train_x_mspec_d, train_y=train_y, valid_x=valid_x_mspec_d, valid_y=valid_y, test_x=test_x_mspec_d, test_y=test_y, input_dim=280)
    dnn(num_layers=4, train_x=train_x_mspec_d, train_y=train_y, valid_x=valid_x_mspec_d, valid_y=valid_y, test_x=test_x_mspec_d, test_y=test_y, input_dim=280)


    # TODO: 5.1 Detailed Evaluation

