from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Padding
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

# Model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=32, input_length=200))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary output

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)


prediction = model.predict(X_test[:5])
print(prediction)