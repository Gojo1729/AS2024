import numpy as np


def hamming_loss(y_true, y_pred):
    # Calculate number of mismatches
    num_mismatches = np.sum(y_true != y_pred)

    # Compute Hamming Loss
    hamming_loss = num_mismatches / (y_true.shape[0] * y_true.shape[1])

    return hamming_loss


def precision(y_true, y_pred, num_classes):
    # Initialize arrays to store true positives, false positives, and precision
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    precision_scores = np.zeros(num_classes)

    # Calculate true positives and false positives for each class
    for i in range(num_classes):
        TP[i] = np.sum((y_true == i) & (y_pred == i))
        FP[i] = np.sum((y_true != i) & (y_pred == i))

    # Compute precision for each class
    for i in range(num_classes):
        if TP[i] + FP[i] > 0:
            precision_scores[i] = TP[i] / (TP[i] + FP[i])

    return np.mean(precision_scores)


class NeuralNetwork:

    def __init__(self, raw_data, embeddings):

        self.raw_data = raw_data
        self.random_state = 42

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.labels = [
            "Joy",
            "Trust",
            "Fear",
            "Surprise",
            "Sadness",
            "Disgust",
            "Anger",
            "Anticipation",
        ]
        self.emotions_onehot = np.array(raw_data.loc[:, self.labels])
        self.__pre_process(embeddings)

        self.n_classes = self.y_train.shape[1]
        self.n_input_features = self.X_train.shape[1]
        self.n_hidden_neurons = 3

        # weights from input layer to hidden layer1
        np.random.seed(self.random_state)
        self.W01 = np.random.randn(self.n_input_features, self.n_hidden_neurons)
        self.W12 = np.random.randn(self.n_hidden_neurons, self.n_classes)

        self.b01 = np.zeros((1, self.n_hidden_neurons))
        self.b12 = np.zeros((1, self.n_classes))

    def __activation(self, activation_function, X):
        if activation_function == "sigmoid":
            return 1 / (1 + np.exp(-X))
        elif activation_function == "relu":
            return (X > 0) * X
        elif activation_function == "tanh":
            return (np.exp(X) + np.exp(-X)) / (np.exp(X) - np.exp(-X))

    def __activation_derivative(self, activation_function, X):
        if activation_function == "sigmoid":
            return self.__activation(activation_function, X) * (
                1 - self.__activation(activation_function, X)
            )
        elif activation_function == "relu":
            return X > 0
        elif activation_function == "tanh":
            return 1 - self.__activation(activation_function, X) ** 2

    def __error(self, preds, ground, error="mean"):
        return -np.mean(ground * np.log(preds) + (1 - ground) * np.log(1 - preds))
        # return 0.5 * preds.shape[1] * ((ground - preds)**2).sum()

    # Note: output activation will always be sigmoid
    def train(
        self,
        epochs=100,
        lr=1e-1,
        hidden_layer_activation="relu",
        batch_size=16,
        thresold=0.6,
    ):
        global log
        log += f"{self.n_hidden_neurons}, {batch_size},"
        log += f"{epochs}, {lr}, {hidden_layer_activation},"
        # forward the data, and then calculate the training accuracy
        self.hidden_layer_activation = hidden_layer_activation
        print(f"number of batches {self.X_train.shape[0]//batch_size}")
        error = 0
        for epoch in range(epochs):
            # batch gd
            batches = self.X_train.shape[0] % batch_size
            exact_batches = True if batches == 0 else False
            n_batches = (
                (self.X_train.shape[0] // batch_size)
                if exact_batches
                else (self.X_train.shape[0] // batch_size + 1)
            )
            for batch in range(n_batches):
                b = batch * batch_size
                b_1 = (
                    self.X_train.shape[0]
                    if (not exact_batches) and (batch == n_batches - 1)
                    else (batch + 1) * batch_size
                )
                self.X_batch = self.X_train[b:b_1]
                self.Y_batch = self.y_train[b:b_1]
                self.Z01 = self.X_batch.dot(self.W01) + self.b01
                self.A01 = self.__activation(hidden_layer_activation, self.Z01)
                self.Z02 = self.A01.dot(self.W12) + self.b12
                self.A02 = self.__activation("sigmoid", self.Z02)

                error = self.__error(self.A02, self.Y_batch)

                self.backward()

                self.W12 -= lr * self.A01.T.dot(self.d_error_W12)
                self.b12 -= lr * np.sum(self.d_error_W12, axis=0, keepdims=True)
                self.W01 -= lr * self.X_batch.T.dot(self.d_error_W01)

            # print(f"Error {error}")
            if epoch % 10 == 0:
                self.test(epoch, self.X_train, self.y_train)
                print(f"Error {error}")
                # train_accuracy = self.accuracy(self.A02, self.Y_batch)
                # print(self.Y_batch, b, b_1)

        # log += f"{train_accuracy},"

    def backward(self):
        self.d_error_A02 = (self.A02 - self.Y_batch) / len(self.Y_batch)
        self.d_error_W12 = (self.d_error_A02) * self.__activation_derivative(
            "sigmoid", self.Z02
        )

        self.d_error_W01 = (self.d_error_W12).dot(
            self.W12.T
        ) * self.__activation_derivative(self.hidden_layer_activation, self.Z01)

    def __pre_process(self, embeddings, train_test_ratio=0.3):
        # self.raw_data = self.raw_data.drop("Id",axis=1)
        # species_np = np.array(self.raw_data["Species"])
        # onehotencoder  = OneHotEncoder(sparse_output = False)
        # target_onehot = onehotencoder.fit_transform(species_np.reshape(-1,1))
        # self.raw_data=self.raw_data.drop("Species", axis=1)

        X = embeddings
        y = self.emotions_onehot

        self.X_train, self.X_test, self.y_train, self.y_test = X, X, y, y

    def accuracy(self, y_pred, y_ground):
        y_train_predicted_classes = np.argmax(y_pred, axis=1)
        y_train_ground_classes = np.argmax(y_ground, axis=1)
        accuracy = ((y_train_predicted_classes == y_train_ground_classes).sum()) / len(
            y_train_predicted_classes
        )
        return accuracy

    def test(self, epoch, X, y):
        global log
        Z01 = X.dot(self.W01) + self.b01
        A01 = self.__activation(self.hidden_layer_activation, Z01)
        Z02 = A01.dot(self.W12) + self.b12
        A02 = self.__activation("sigmoid", Z02)
        predictions = A02.round()
        precision_metric = precision(y, predictions, y.shape[1])
        hamming_loss_metric = hamming_loss(y, predictions)
        print(
            f"Epoch {epoch}, Precision {precision_metric}, Hamming loss {hamming_loss_metric}"
        )

    def predict(self, X):
        Z01 = X.dot(self.W01) + self.b01
        A01 = self.__activation(self.hidden_layer_activation, Z01)
        Z02 = A01.dot(self.W12) + self.b12
        A02 = self.__activation("sigmoid", Z02)
        return A02

    def print_shapes(self):
        print(f"Xtrain shape {self.X_train.shape}")
        print(f"ytrain shape {self.y_train.shape}")
        print(f"Xtest shape {self.X_test.shape}")
        print(f"ytest shape {self.y_test.shape}")
