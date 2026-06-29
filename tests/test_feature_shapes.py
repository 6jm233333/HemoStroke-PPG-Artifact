import pytest

torch = pytest.importorskip("torch")

from src.models.lstm import LSTMClassifier
from src.models.resnet1d import resnet1d18


def test_resnet1d_accepts_btc_feature_sequences():
    model = resnet1d18(input_dim=17, output_dim=2, input_layout="btc")
    x = torch.randn(4, 500, 17)
    y = model(x)
    assert tuple(y.shape) == (4, 2)


def test_lstm_baseline_accepts_btc_feature_sequences():
    model = LSTMClassifier(input_dim=17, output_dim=2, hidden_dim=16, num_layers=1)
    x = torch.randn(4, 500, 17)
    y = model(x)
    assert tuple(y.shape) == (4, 2)
