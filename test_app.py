import pytest
from app import app
from bs4 import BeautifulSoup

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict(client):
    response = client.post('/predict', data={'text': 'pytest text'})
    
    # Test endpoint
    assert response.status_code == 200

    # Parse HTML content returned by the server
    soup = BeautifulSoup(response.data, 'html.parser')

    # Extract values passed to the index.html template
    predictions = [float(p['style'].split(':')[1].strip('%'))/100 for p in soup.find_all('div', {'class': 'progress'})]

    # Test predictions
    assert len(predictions) > 0
