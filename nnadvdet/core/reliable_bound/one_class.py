class OneClassDetector:
    def __init__(self, config):
        self.config = config

    def train(self, x_train, pred_train, label_train, weights=None):
        raise Exception('Not Implemented.')

    def predict(self, x, x_pred):
        raise Exception('Not Implemented.')

    def batched_predict(self, x, x_pred):
        raise Exception('Not Implemented.')
