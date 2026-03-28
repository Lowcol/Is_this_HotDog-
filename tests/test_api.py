import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

import numpy as np

# Import the FastAPI app
import main
from main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_no_model(self):
        """Test health endpoint when model isn't loaded."""
        with patch("main.model", None):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "healthy")
            self.assertFalse(response.json()["model_loaded"])

    def test_health_check_with_model(self):
        """Test health endpoint when model is loaded."""
        with patch("main.model", MagicMock()):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "healthy")
            self.assertTrue(response.json()["model_loaded"])

    def test_predict_no_model(self):
        """Test predict endpoint when model isn't loaded returns 503."""
        with patch("main.model", None):
            # Create a dummy image
            file_payload = {"file": ("test.jpg", b"dummy content", "image/jpeg")}
            response = self.client.post("/predict", files=file_payload)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"], "Model not loaded")

    def test_predict_success(self):
        """Test a successful prediction with a mocked model."""
        mock_model = MagicMock()
        # predict returns a numpy array with shape (1, 2)
        mock_model.predict.return_value = np.array([[0.1, 0.9]]) 

        with patch("main.model", mock_model):
            # Create a valid tiny image header using PIL
            img = Image.new('RGB', (10, 10), color='red')
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            file_payload = {"file": ("test.jpg", img_byte_arr.read(), "image/jpeg")}
            response = self.client.post("/predict", files=file_payload)
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["prediction"], "hotdog")
            self.assertAlmostEqual(data["confidence"], 0.9)
            
    def test_predict_invalid_image(self):
        """Test prediction with an invalid file format."""
        mock_model = MagicMock()
        with patch("main.model", mock_model):
            # Pass a text file instead of an image
            file_payload = {"file": ("test.txt", b"this is text, not an image", "text/plain")}
            response = self.client.post("/predict", files=file_payload)
            
            self.assertEqual(response.status_code, 400)
            self.assertIn("cannot identify image file", response.json()["detail"])

if __name__ == '__main__':
    unittest.main()
