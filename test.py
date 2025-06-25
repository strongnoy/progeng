import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200

def test_detect_with_valid_image():
    with open("test_images/test.jpg", "rb") as img_file:
        files = {"file": ("test.jpg", img_file, "image/jpeg")}
        response = client.post("/detect/", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] in ["image/png", "image/jpeg"]
    assert len(response.content) > 0


def test_detect_with_invalid_file():
    files = {"file": ("test.txt", b"This is not an image", "text/plain")}
    response = client.post("/detect/", files=files)

    assert response.status_code == 422 or response.status_code == 400


def test_detect_without_file():
    response = client.post("/detect/", files={})
    assert response.status_code == 422
