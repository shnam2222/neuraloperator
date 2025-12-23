# Set variables in this file

################################################################
#  configurations
################################################################   
'''       
n_total = 23561
n_train = 10000
n_test = 2000
epochs = 500
batch_size = 200
learning_rate = 0.001
step_size = 100
gamma = 0.5
size  = 256
'''

class config():
    def __init__(self):
        self.n_total = 23561
        self.n_train = 10000
        self.n_test = 2000
        self.epochs = 500
        self.batch_size = 200
        self.learning_rate = 0.001
        self.step_size = 100
        self.gamma = 0.5
        self.size  = 256
    
    def update_n_total(self, n_total):
        self.n_total = n_total
    def update_n_train(self, n_train):
        self.n_train = n_train
    def update_n_test(self, n_test):
        self.n_test = n_test
    def update_epochs(self, epochs):
        self.epochs = epochs
    def update_batch_size(self, batch_size):
        self.batch_size = batch_size
    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
    def update_step_size(self, step_size):
        self.step_size = step_size
    def update_gamma(self, gamma):
        self.gamma = gamma
    def update_size(self, size):
        self.size = size
        
instance = config()