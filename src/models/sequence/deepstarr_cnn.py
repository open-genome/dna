### Dictionary containing various hyperparameters for the model, to be more flexible to change them and test different model architectures

# These are the parameters from the main model trained using genome-wide data - more details at https://github.com/bernardo-de-almeida/DeepSTARR/tree/main/DeepSTARR
params_full_gw_data = {'batch_size': 128,
                      'epochs': 100,
                      'early_stop': 5,
                      'lr': 0.002,
                      'n_conv_layer': 4,
                      'num_filters1': 256,
                      'num_filters2': 60,
                      'num_filters3': 60,
                      'num_filters4': 120,
                      'kernel_size1': 7,
                      'kernel_size2': 3,
                      'kernel_size3': 5,
                      'kernel_size4': 3,
                      'n_dense_layer': 2,
                      'dense_neurons1': 256,
                      'dense_neurons2': 256,
                      'dropout_conv': 'no',
                      'dropout_prob': 0.4,
                      'pad':'same'}


# Here we will adapt the model architecture for the training with the smaller data
# note the reduction in the number of layers, filters, neurons, addition of dropout on the convolutional layers
# this model has 165,354 parameters compared with 624,738 from the full model above
# params_smaller_data = {'batch_size': 64, # number of examples per batch
#                       'epochs': 100, # number of epochs
#                       'early_stop': 10, # patience of 10 epochs to reduce training time; you can increase the patience to see if the model improves after more epochs
#                       'lr': 0.001, # learning rate
#                       'n_conv_layer': 3, # number of convolutional layers
#                       'num_filters1': 128, # number of filters/kernels in the first conv layer
#                       'num_filters2': 60, # number of filters/kernels in the second conv layer
#                       'num_filters3': 60, # number of filters/kernels in the third conv layer
#                       # 'num_filters4': 120,
#                       'kernel_size1': 7, # size of the filters in the first conv layer
#                       'kernel_size2': 3, # size of the filters in the second conv layer
#                       'kernel_size3': 5, # size of the filters in the third conv layer
#                       # 'kernel_size4': 3,
#                       'n_dense_layer': 1, # number of dense/fully connected layers
#                       'dense_neurons1': 64, # number of neurons in the dense layer
#                       # 'dense_neurons2': 256,
#                       'dropout_conv': 'yes', # add dropout after convolutional layers?
#                       'dropout_prob': 0.4, # dropout probability
#                       'pad':'same'}

# def DeepSTARR(params):

#     # expects sequences of length 249 with 4 channels, length of DNA sequences
#     input = kl.Input(shape=(249, 4))

#     # Body - 4 conv + batch normalization + ReLu activation + max pooling
#     # The number of convolutional layers and their hyperparameters are determined by the values in the params dictionary.
#     x = kl.Conv1D(params['num_filters1'], kernel_size=params['kernel_size1'],
#                   padding=params['pad'],
#                   name='Conv1D_1')(input)
#     x = kl.BatchNormalization()(x)
#     x = kl.Activation('relu')(x)
#     x = kl.MaxPooling1D(2)(x)

#     for i in range(1, params['n_conv_layer']):
#         x = kl.Conv1D(params['num_filters'+str(i+1)],
#                       kernel_size=params['kernel_size'+str(i+1)],
#                       padding=params['pad'],
#                       name=str('Conv1D_'+str(i+1)))(x)
#         x = kl.BatchNormalization()(x)
#         x = kl.Activation('relu')(x)
#         x = kl.MaxPooling1D(2)(x)
#         # add dropout after convolutional layers?
#         if params['dropout_conv'] == 'yes': x = kl.Dropout(params['dropout_prob'])(x)

#     # After the convolutional layers, the output is flattened and passed through a series of fully connected/dense layers
#     # Flattening converts a multi-dimensional input (from the convolutions) into a one-dimensional array (to be connected with the fully connected layers
#     x = kl.Flatten()(x)

#     # Fully connected layers
#     # Each fully connected layer is followed by batch normalization, ReLU activation, and dropout
#     for i in range(0, params['n_dense_layer']):
#         x = kl.Dense(params['dense_neurons'+str(i+1)],
#                      name=str('Dense_'+str(i+1)))(x)
#         x = kl.BatchNormalization()(x)
#         x = kl.Activation('relu')(x)
#         x = kl.Dropout(params['dropout_prob'])(x)

#     # Main model bottleneck
#     bottleneck = x

#     # heads per task (developmental and housekeeping enhancer activities)
#     # The final output layer is a pair of dense layers, one for each task (developmental and housekeeping enhancer activities), each with a single neuron and a linear activation function
#     tasks = ['Dev', 'Hk']
#     outputs = []
#     for task in tasks:
#         outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(bottleneck))

#     # Build Keras model object
#     model = Model([input], outputs)
#     model.compile(Adam(learning_rate=params['lr']), # Adam optimizer
#                   loss=['mse', 'mse'], # loss is Mean Squared Error (MSE)
#                   loss_weights=[1, 1]) # in case we want to change the weights of each output. For now keep them with same weights

#     return model, params

import torch
from torch import nn
from torch.optim import Adam

class DeepSTARR(nn.Module):
    def __init__(self, params_full_gw_data):
        super().__init__()
        self.params = params_full_gw_data
        params = params_full_gw_data

        # Convolutional layers
        self.convs = nn.ModuleList()
        for i in range(params['n_conv_layer']):
            self.convs.append(nn.Sequential(
                nn.Conv1d(5 if i == 0 else params['num_filters'+str(i)], params['num_filters'+str(i+1)], kernel_size=params['kernel_size'+str(i+1)], padding=int(params['kernel_size'+str(i+1)]/2)),
                nn.BatchNorm1d(params['num_filters'+str(i+1)]),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(params['dropout_prob']) if params['dropout_conv'] == 'yes' else nn.Identity()
            ))

        # Dense layers
        self.denses = nn.ModuleList()
        for i in range(params['n_dense_layer']):
            self.denses.append(nn.Sequential(
                nn.Linear(params['dense_neurons'+str(i)], params['dense_neurons'+str(i+1)]),
                nn.BatchNorm1d(params['dense_neurons'+str(i+1)]),
                nn.ReLU(),
                nn.Dropout(params['dropout_prob'])
            ))

        # Output layers
        self.out_dev = nn.Linear(params['dense_neurons'+str(params['n_dense_layer'])], 2)
        # self.out_hk = nn.Linear(params['dense_neurons'+str(params['n_dense_layer'])], 1)

    def forward(self, seq, t=None, cls = None, return_embedding=False, state=None):
        x = torch.nn.functional.one_hot(seq, num_classes=self.alphabet_size).type(torch.float32)
        for conv in self.convs:
            x = conv(x)
        x = x.view(x.size(0), -1)
        for dense in self.denses:
            x = dense(x)
        out_dev = self.out_dev(x)
        # out_hk = self.out_hk(x)
        return out_dev, None
    
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model
