import torch
import torch.nn as nn
from mlops.model import Model


def test_single_training_step():
    model = Model(output_dim=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 8, (2,))

    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"
