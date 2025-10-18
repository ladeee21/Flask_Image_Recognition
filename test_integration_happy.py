# test_integration_happy.py

"""Integration tests for happy path scenarios in image upload and prediction."""
from io import BytesIO

def simulate_image_upload(client, image_bytes, filename="test_image.jpg"):
    """Helper to simulate an image upload to the /prediction endpoint."""
    img_data = BytesIO(image_bytes)
    img_data.name = filename

    response = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )
    return response

def test_successful_prediction(client):
    """Test the successful image upload and prediction."""
    response = simulate_image_upload(client, b"fake_image_data", "test.jpg")
    assert response.status_code == 200
    assert b"Prediction" in response.data
