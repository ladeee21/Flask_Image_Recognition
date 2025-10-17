# test_model.py

"""Unit tests for model functions: preprocess_img and predict_result."""
import pytest
import numpy as np
from keras.models import load_model
from model import preprocess_img, predict_result
from PIL import Image
import os

# Load the model before tests run
@pytest.fixture(scope="module")
def model():
    """Load the model once for all tests."""
    model = load_model("digit_model.h5")
    return model

# BASIC TESTS
def test_preprocess_img():
    """Test the preprocess_img function."""
    img_path = "test_images/2/Sign 2 (97).jpeg"
    processed_img = preprocess_img(img_path)
    assert processed_img.shape == (1, 224, 224, 3), "Processed image shape should be (1, 224, 224, 3)"
    assert np.min(processed_img) >= 0 and np.max(processed_img) <= 1, "Pixel values should be normalized"

def test_predict_result(model):
    """Test the predict_result function."""
    img_path = "test_images/4/Sign 4 (92).jpeg"
    processed_img = preprocess_img(img_path)
    prediction = predict_result(processed_img)
    assert isinstance(prediction, (int, np.integer)), "Prediction should be an integer class index"

# ADVANCED TESTS

def test_invalid_image_path():
    """Test with invalid path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("invalid/path/to/image.jpeg")

def test_image_shape_on_prediction(model):
    """Test prediction on another image."""
    img_path = "test_images/5/Sign 5 (86).jpeg"
    processed_img = preprocess_img(img_path)
    prediction = predict_result(processed_img)
    assert isinstance(prediction, (int, np.integer)), "The prediction should be an integer"

def test_model_predictions_consistency(model):
    """Test predictions are consistent for the same image."""
    img_path = "test_images/7/Sign 7 (54).jpeg"
    processed_img = preprocess_img(img_path)
    predictions = [predict_result(processed_img) for _ in range(5)]
    assert all(p == predictions[0] for p in predictions), "Predictions should be consistent"

# NEW TESTS (Part 2)

def test_blurry_image_prediction(model):
    """Test prediction on a blurry image (if available)."""
    img_path = "test_images/blurry/Sign_blur_1.jpeg"
    if os.path.exists(img_path):
        processed_img = preprocess_img(img_path)
        prediction = predict_result(processed_img)
        assert isinstance(prediction, (int, np.integer))

def test_partial_sign_image(model):
    """Test image that only partially shows a sign."""
    img_path = "test_images/partial/Sign_partial.jpeg"
    if os.path.exists(img_path):
        processed_img = preprocess_img(img_path)
        prediction = predict_result(processed_img)
        assert isinstance(prediction, (int, np.integer))

def test_wrong_file_type():
    """Pass non-image file to preprocess_img and expect error."""
    with pytest.raises(OSError):
        preprocess_img("test_images/invalid_text.txt")

def test_blank_image_prediction(model):
    """Predict result on a blank white image."""
    # Create a blank white image in memory
    img = Image.new("RGB", (224, 224), color="white")
    img_arr = np.array(img) / 255.0
    img_input = img_arr.reshape(1, 224, 224, 3)
    prediction = predict_result(img_input)
    assert isinstance(prediction, (int, np.integer)), "Should return class even for blank image"
