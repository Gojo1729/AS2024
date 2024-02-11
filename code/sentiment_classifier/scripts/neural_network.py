# class to be imported in the code/model_interface while loading the pickle file
import numpy as np


def top3_accuracy(predicted_probs, true_labels):
    """
    Calculates the top-3 accuracy.

    Args:
    - predicted_probs (np.ndarray): Predicted probabilities for each class.
    - true_labels (np.ndarray): True labels for each sample.

    Returns:
    - float: Top-3 accuracy score.

    """

    sorted_indices = np.argsort(predicted_probs, axis=1)[:, ::-1]

    # Check if true labels are in top-3 predicted labels
    top3_correct = np.any(true_labels[np.arange(len(true_labels))[:, None], sorted_indices[:, :3]], axis=1)
    # Calculate top-3 accuracy
    top3_accuracy = np.mean(top3_correct)
    
    return top3_accuracy

def hamming_loss(y_true, y_pred):
    """
    Computes the Hamming loss.

    Args:
    - y_true (np.ndarray): True labels.
    - y_pred (np.ndarray): Predicted labels.

    Returns:
    - float: Hamming loss.

    """
    # Calculate number of mismatches
    num_mismatches = np.sum(y_true != y_pred)

    # Compute Hamming Loss
    hamming_loss = num_mismatches / (y_true.shape[0] * y_true.shape[1])

    return hamming_loss


def precision(y_true, y_pred, num_classes):
    """
    Computes the precision.

    Args:
    - y_true (np.ndarray): True labels.
    - y_pred (np.ndarray): Predicted labels.
    - num_classes (int): Number of classes.

    Returns:
    - float: Mean precision score.

    """
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
    
    def __init__(self, raw_data, embeddings, hidden_neurons, test_df, test_embeddings):
        
        self.raw_data = raw_data
        self.random_state = 42
        np.random.seed(self.random_state)
        
        self.labels = ['Joy', 'Trust', 'Fear', 'Surprise','Sadness', 'Disgust', 'Anger', 'Anticipation']
        self.emotions_onehot = np.array(raw_data.loc[:, self.labels])
        self.X_train, self.y_train = embeddings, self.emotions_onehot
        self.X_test, self.y_test = test_embeddings, np.array(test_df.loc[:, self.labels])
        
        self.n_classes = self.y_train.shape[1]
        self.n_input_features = self.X_train.shape[1]
        self.n_hidden_neurons = hidden_neurons
        
        
        #weights from input layer to hidden layer1
        limit1 = np.sqrt(2 / float(self.n_input_features + self.n_hidden_neurons))
        limit2 = np.sqrt(2 / float(self.n_hidden_neurons + self.n_classes))

        self.W01 = np.random.normal(0.0, limit1, size=(self.n_input_features, self.n_hidden_neurons))
        self.W12 = np.random.normal(0.0, limit2, size=(self.n_hidden_neurons, self.n_classes))
         
        self.b01 = np.zeros((1, self.n_hidden_neurons))
        self.b12 = np.zeros((1, self.n_classes))
        
    def __activation(self, activation_function, X) -> np.ndarray:
        if activation_function == "sigmoid":
            return 1/(1+np.exp(-X))
        elif activation_function == "relu":
            return (X > 0) * X
        elif activation_function == "tanh":
            return (np.exp(X) + np.exp(-X))/(np.exp(X) - np.exp(-X))
        else:
            raise Exception(f"Activation function {activation_function} not defined")
        
    def __activation_derivative(self, activation_function, X) -> np.ndarray:
        if activation_function == "sigmoid":
            return self.__activation(activation_function, X) * (1 - self.__activation(activation_function, X))
        elif activation_function == "relu":
            return X > 0
        elif activation_function == "tanh":
            return 1 - self.__activation(activation_function, X)**2
        else:
            raise Exception(f"Activation function {activation_function} not defined")
    
    def __error(self, preds, ground, error="mean"):
        return -np.mean(ground * np.log(preds) + (1 - ground) * np.log(1 - preds))
        
    
    #Note: output activation will always be sigmoid
    def train(self, epochs=100, lr = 1e-1, hidden_layer_activation = "relu", batch_size = 16, thresold = 0.6):
        # forward the data, and then calculate the training accuracy
        self.hidden_layer_activation = hidden_layer_activation
        print(f"number of batches {self.X_train.shape[0]//batch_size}")
        train_error = 0
        for epoch in range(epochs):
            #batch gd
            batches = (self.X_train.shape[0] % batch_size)
            exact_batches = True if batches == 0 else False
            n_batches = (self.X_train.shape[0]//batch_size) if exact_batches else (self.X_train.shape[0]//batch_size + 1)
            for batch in range(n_batches):
                b = batch*batch_size
                b_1 = self.X_train.shape[0] if (not exact_batches) and (batch == n_batches-1) else (batch+1)*batch_size
                self.X_batch = self.X_train[b:b_1]
                self.Y_batch = self.y_train[b:b_1]
                self.Z01 = self.X_batch.dot(self.W01) + self.b01
                self.A01 = self.__activation(hidden_layer_activation, self.Z01)
                self.Z02 = self.A01.dot(self.W12) + self.b12
                self.A02 = self.__activation("sigmoid", self.Z02)
                
                train_error = self.__error(self.A02, self.Y_batch)

                self.backward()

                self.W12 -= lr * self.A01.T.dot(self.d_error_W12)
                self.b12 -= lr * np.sum(self.d_error_W12, axis=0, keepdims = True)
                self.W01 -= lr * self.X_batch.T.dot(self.d_error_W01)

            if epoch % 10 == 0:
                test_error, precision_metric, hamming_loss_metric, top3_metric = self.test(epoch)
                print(f"Epoch {epoch}, Train error {train_error}, Test error {test_error}, Precision {precision_metric}, top3metric {top3_metric}, Hamming loss {hamming_loss_metric}")

                
    def backward(self):
        self.d_error_A02 = (self.A02 - self.Y_batch)/len(self.Y_batch)
        self.d_error_W12 = (self.d_error_A02) * self.__activation_derivative("sigmoid", self.Z02)
        
        self.d_error_W01 = (
            (self.d_error_W12).dot(self.W12.T) * self.__activation_derivative(self.hidden_layer_activation, self.Z01))
    
    def accuracy(self, y_pred, y_ground):
        y_train_predicted_classes = np.argmax(y_pred, axis = 1)
        y_train_ground_classes = np.argmax(y_ground, axis=1)
        accuracy = ((y_train_predicted_classes == y_train_ground_classes).sum())/len(y_train_predicted_classes)
        return accuracy
               
    def test(self, epoch):
        X, y = self.X_test, self.y_test
        Z01 = X.dot(self.W01) + self.b01
        A01 = self.__activation(self.hidden_layer_activation, Z01)
        Z02 = A01.dot(self.W12) + self.b12
        A02 = self.__activation("sigmoid", Z02)
        predictions = A02.round()
        error = self.__error(A02, y)
        precision_metric = precision(y, predictions, y.shape[1])
        hamming_loss_metric = hamming_loss(y, predictions)
        top3_metric = top3_accuracy(A02, y)
        return error, precision_metric, hamming_loss_metric, top3_metric


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
    
        