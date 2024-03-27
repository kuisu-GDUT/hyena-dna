from torch import nn

# A simple CNN model
from src.models.sequence import SequenceModule, TransposedModule


# @TransposedModule
class CNN(SequenceModule):
    def __init__(self, d_input, d_output=None, **kwargs):
        super(CNN, self).__init__()

        self.d_output = d_input if d_output is None else d_output
        self.cnn_model = nn.Sequential(
            nn.Conv1d(in_channels=d_input, out_channels=self.d_output, kernel_size=5, bias=True, padding=2),
            nn.BatchNorm1d(self.d_output),
            nn.ReLU(),
            # nn.MaxPool1d(2),

            # nn.Conv1d(in_channels=self.d_output, out_channels=self.d_output, kernel_size=2, bias=True),
            # nn.BatchNorm1d(self.d_output),
            # # nn.MaxPool1d(2),
            #
            # nn.Conv1d(in_channels=self.d_output, out_channels=self.d_output, kernel_size=2, bias=True),
            # nn.BatchNorm1d(self.d_output),
            # nn.MaxPool1d(2),

            # nn.Flatten()
        )

    def forward(self, x, *args, **kwargs):
        x = x.transpose(-1, -2)
        x = self.cnn_model(x)
        x = x.transpose(-1, -2)
        return x

    def step(self, x, state=None, **kwargs):
        return self.cnn_model(x), state
