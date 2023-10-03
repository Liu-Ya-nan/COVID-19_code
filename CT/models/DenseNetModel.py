import torch.nn as nn
import torchxrayvision as xrv


class DenseNetModel(nn.Module):

    def __init__(self):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        super(DenseNetModel, self).__init__()

        self.dense_net = xrv.models.DenseNet(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.dense_net(x)
        return logits

def get_model(**kwargs):
    """
    Returns the model.
    """
    model = DenseNetModel(**kwargs)
    return model
