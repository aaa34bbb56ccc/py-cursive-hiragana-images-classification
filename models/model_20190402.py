# Model
model = Sequential()

# First CNN layer
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', padding="same",
        kernel_initializer='he_normal', 
        input_shape=(IMG_ROWS, IMG_COLS, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same",
        kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', padding="same",
        kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.15))

# 2nd CNN layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same",
        kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu',padding="same",
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0002)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# 3nd CNN layer
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same",
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0002)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), activation='relu',padding="same",
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0003)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# 4nd CNN layer
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same",
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0003)))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(5, 5), strides=(2, 2), activation='relu',padding="same",
        kernel_initializer='he_normal', kernel_regularizer=l2(0.0004)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# fully-connected layer
model.add(Flatten())
model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# softmax classifier
model.add(Dense(NUM_CLASSES, activation='softmax'))