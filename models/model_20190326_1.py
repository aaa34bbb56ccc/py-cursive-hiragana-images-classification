# Model
model = Sequential()
# Add convolution 2D
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu', padding="same",
        kernel_initializer='he_normal',kernel_regularizer=l2(0.0005), 
        input_shape=(IMG_ROWS, IMG_COLS, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same",
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# stack two more CONV layers, keeping the size of each filter
# as 3x3 but increasing to 64 total learned filters
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same",
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu',padding="same",
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# increase the number of filters again, this time to 128
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding="same",
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu',padding="same",
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# fully-connected layer
model.add(Flatten())
model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(NUM_CLASSES, activation='softmax'))