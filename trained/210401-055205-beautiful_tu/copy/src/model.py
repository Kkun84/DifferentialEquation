from torch import nn, Tensor


class Model(nn.Module):
    def __init__(self, input_n: int, output_n: int, hidden_n: int) -> None:
        super().__init__()

        self.input_shape = (input_n,)
        self.output_shape = (output_n,)

        self.hidden_n = hidden_n

        self.acctivate = nn.Softplus()

        self.fc1 = nn.Linear(input_n, self.hidden_n)
        self.fc2 = nn.Linear(self.hidden_n, self.hidden_n)
        self.fc3 = nn.Linear(self.hidden_n, output_n)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.acctivate(x)
        x = self.fc2(x)
        x = self.acctivate(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    model = Model(1, 1, 64)
    summary(model)
