from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import math


class Regressor(nn.Module):

    def __init__(self, x, y=None, batch_size=10, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - batch_size {int} -- The batch size that will be used for training.
            - nb_epoch {int} -- Number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        super(Regressor, self).__init__()
        X, _ = self._preprocessor(x, y, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self._nn = nn.Sequential(
            nn.Linear(self.input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, self.output_size)
        )
        self.optimiser = optim.SGD(self._nn.parameters(), lr=0.01, momentum=0.9)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        return self._nn(x)    
    
    def _fill_nans(self, x, training):
        """
        Fills all the columns that contain NaNs with the mean values of the
        respective columns.
        
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.
        
        Returns:
            - {pd.DataFrame} -- Input array with missing values filled
        """
        
        # If training is True then find the columns with NaNs, find the 
        # median of those columns and then store them
        if training:
            self._columns_with_nans = x.columns[x.isna().any()]
            self._means = (x[self._columns_with_nans]
                                .mean())
        
        x.loc[:, self._columns_with_nans] = (x[self._columns_with_nans]
                                                .fillna(self._means))
        
        return x
    
    def _normalise_columns(self, x, training, y=None):
        """
        Normalise each numeric column so that they all have values between 0 and
        1.
        
        Arguments:
            - df {pd.DataFrame} -- Partially pre-processed input array of shape 
                (batch_size, input_size).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
        
        Returns:
            - {pd.DataFrame} -- Input array with normalised numeric values
        """
        
        # If training is True then find all of the numeric columns and fit the
        # sklearn MinMaxScaler to them. Store the scaler for later use.
        if training:
            self._normalised_columns = x.select_dtypes(include="number").columns
            self._x_min_max_scaler = MinMaxScaler()
            self._x_min_max_scaler.fit(x[self._normalised_columns])
            
            # If y is provided then store the scaler for normalising y
            if y is not None:
                self._y_min_max_scaler = MinMaxScaler()
                self._y_min_max_scaler.fit(y)
        
        x.loc[:, self._normalised_columns] = (self._x_min_max_scaler.transform(
                                                x[self._normalised_columns]
                                             ))
        
        return x, (self._y_min_max_scaler.transform(y) 
                   if isinstance(y, pd.DataFrame) 
                   else None)

    def _one_hot_encoding(self, x, training):
        """
        Apply one hot encoding onto each column that contains categorical data.
        
        Arguments:
            - df {pd.DataFrame} -- Partially pre-processed input array of shape 
                (batch_size, input_size).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.
        
        Returns:
            - {pd.DataFrame} -- Input array with one hot encoded columns
        """
        
        # If training is true then find all categorical columns and fit the 
        # sklearn OneHotEncoder to them. Store the encoder for later use.
        if training:
            self._categorical_columns = (x.select_dtypes(include=[object])
                                            .columns)
            self._one_hot_encoder = OneHotEncoder()
            self._one_hot_encoder.fit(x[self._categorical_columns])
        
        # Find the one hot encoding of the categorical data
        encoding = pd.DataFrame(
            self._one_hot_encoder.transform(x[self._categorical_columns])
                .toarray(), 
            columns=self._one_hot_encoder.categories_
        )
        
        # Concatenate the encoded columns with the input array with the original
        # categorical columns dropped
        x = pd.concat([x, encoding], axis=1).drop(self._categorical_columns, 
                                                  axis=1)
        
        return x

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be 
              the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x_copy = x.copy()
        y_copy = y.copy() if y is not None else None

        x_without_nans = self._fill_nans(x_copy, training)
        normalised_x, normalised_y = self._normalise_columns(x_without_nans, 
                                                             training, y_copy)
        encoded_x = self._one_hot_encoding(normalised_x, training)
        return encoded_x, (normalised_y 
                              if isinstance(y, pd.DataFrame) 
                              else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y) # Do not forget
        
        input_tensor = torch.tensor(X.values, dtype=torch.float32)
        target_tensor = torch.tensor(Y, dtype=torch.float32)

        input_tensor = input_tensor.to(torch.device("cuda"))
        target_tensor = target_tensor.to(torch.device("cuda"))

        print(input_tensor.dtype)
        print(target_tensor.dtype)
        # input_tensor = tf.convert_to_tensor(X)
        # target_tensor = tf.convert_to_tensor(Y)
        
        dataset = utils.data.TensorDataset(input_tensor, target_tensor)


        # Shuffles dataset and splits into batches
        shuffled_batch = utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


        for epoch in range(self.nb_epoch):
            for batch_input, batch_target in shuffled_batch:
                batch_input, batch_target = batch_input.to(torch.device("cuda")), batch_target.to(torch.device("cuda"))
                self.optimiser.zero_grad()
                #Forward pass
                predictions = self.forward(batch_input)
                
                loss = self.criterion(predictions, batch_target)
                # Backward pass
                loss.backward()

                # One step gradient descent
                self.optimiser.step()
        
            print(f"Epoch: {epoch}\t L: {loss}")
        
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """
        # batch = x
        # for linear, activation in self._layers:
        #     batch = linear.forward(batch)
        #     if activation is not None:
        #         batch = activation.forward(batch)

        # return batch
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        X_tensor = torch.tensor(X.values, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.forward(X_tensor)

        return predictions.numpy()
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget

        # Convert to PyTorch tensors
        
        input_tensor = torch.tensor(X.values, dtype=torch.float32)
        target_tensor = torch.tensor(Y, dtype=torch.float32)

        input_tensor = input_tensor.to(torch.device("cuda"))
        target_tensor = target_tensor.to(torch.device("cuda"))

        # Forward pass to get predictions
        predictions = self.forward(input_tensor)

        # Calculate Mean Squared Error
        mse = nn.MSELoss()(predictions, target_tensor)

        return math.sqrt(mse.item())

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU instead.")

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    
    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting

    regressor = Regressor(x_train, y=y_train, batch_size = 8, nb_epoch = 10)##
    regressor.to(device)
    regressor.fit(x_train, y_train)
    
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))
    
def test_preprocessor():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    
    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting

    regressor = Regressor(x_train, y=y_train, batch_size = 8, nb_epoch = 1000)

    # x, y = regressor._preprocessor(x_train, y_train, training=False)

if __name__ == "__main__":
    example_main()
    # test_preprocessor()

