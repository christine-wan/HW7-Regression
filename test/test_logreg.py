"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import LogisticRegressor
from regression import loadDataset


# Load dataset with additional features
X_train, X_val, y_train, y_val = loadDataset(
    features=[
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS',
        'Diastolic Blood Pressure',
        'Body Mass Index',
        'Body Weight'
    ],
    split_percent=0.8,
    split_seed=42
)

# Initialize model
num_features = X_train.shape[1]
model = LogisticRegressor(num_feats=num_features, learning_rate=1e-5, tol=1e-5, max_iter=200, batch_size=20)


def test_prediction():
    """Ensure predictions are valid and correctly sized."""

    # Initialize model weights to 0 & make predictions
    model.W = np.zeros(num_features + 1)
    y_pred = model.make_prediction(np.hstack([X_train, np.ones((X_train.shape[0], 1))]))

    # Ensure number of predictions matches number of input samples
    assert len(y_pred) == X_train.shape[0], "Predictions should match number of samples."

    # Check that all predictions are within valid probability range [0, 1]
    assert np.all((y_pred >= 0) & (y_pred <= 1)), "Predictions should be between 0 and 1."

    # with zero weights -> should output 0.5
    assert np.allclose(y_pred, 0.5, atol=1e-5), "Expected all predictions to be 0.5 with zero weights."


def test_loss_function():
    """Ensure the loss function behaves correctly."""

    # Define sample true labels and predicted probabilities
    y_true = np.array([1, 0])
    y_pred = np.array([0.95, 0.05])

    # Check loss for perfect predictions is close to zero
    loss_perfect = model.loss_function(y_true, y_true)
    assert np.isclose(loss_perfect, 0, atol=1e-5), "Loss should be near zero for perfect predictions."

    # Manually compute loss
    epsilon = 1e-6
    loss_manual = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

    # Compute loss
    model_loss = model.loss_function(y_true, y_pred)

    assert np.isclose(loss_manual, model_loss, atol=1e-5), f"Manual loss: {loss_manual}, Model loss: {model_loss}"


def test_gradient():
    """Check if the gradient computation is valid."""

    # Initialize model weights to zero
    model.W = np.zeros(num_features + 1)

    X_padded = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

    # Compute manual predictions & error
    y_pred_manual = model.sigmoid(np.dot(X_padded, model.W))
    error = y_pred_manual - y_train

    # Compute gradient manually & with model
    manual_gradient = np.dot(X_padded.T, error) / len(y_train)
    gradient = model.calculate_gradient(y_train, X_padded)

    assert gradient.shape == model.W.shape, "Gradient shape mismatch."
    assert np.all(np.isfinite(gradient)), "Gradient contains NaN or infinite values."
    assert np.any(gradient != 0), "Gradient should not be all zeros."
    assert np.allclose(gradient, manual_gradient, atol=1e-5), "Manual and model gradient should match."


def test_training():
    """Ensure weights update and loss decreases during training."""

    # Initialize model weights to zero
    model.W = np.zeros(num_features + 1)
    W_start = model.W.copy()

    # Compute initial loss before training
    initial_loss = model.loss_function(y_val, model.make_prediction(np.hstack([X_val, np.ones((X_val.shape[0], 1))])))

    # Model training
    model.train_model(X_train, y_train, X_val, y_val)
    W_end = model.W.copy()
    final_loss = model.loss_function(y_val, model.make_prediction(np.hstack([X_val, np.ones((X_val.shape[0], 1))])))

    assert not np.allclose(W_start, W_end), "Weights should change after training."
    assert final_loss < initial_loss, "Loss should decrease after training."


# --- Exception Handling Tests ---
def test_invalid_calculate_gradient_nan():
    """Test if gradient computation raises an error when X contains NaN values."""
    model = LogisticRegressor(num_feats=num_features)

    X_invalid = np.random.rand(10, num_features)
    X_invalid[2, 1] = np.nan  # Introduce NaN value

    with pytest.raises(ValueError, match="`X` must be non-empty and not contain NaN values."):
        model.calculate_gradient(y_train[:10], X_invalid)


def test_invalid_loss_function_nan():
    """Test if loss function raises an error when y_true contains non-numeric values."""
    y_true_invalid = np.array(["a", "b", "c"], dtype=object)  # Non-numeric values
    y_pred_valid = np.array([0.9, 0.1, 0.8])

    with pytest.raises(ValueError, match="`y_true` should contain numeric data only."):
        model.loss_function(y_true_invalid, y_pred_valid)
