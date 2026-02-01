import pytest
from src.categorize import predict_category

def test_category_security():
    requirement = "The login page must block IP addresses after 5 failed attempts."
    category = predict_category(requirement)
    assert category == "Security"

def test_category_performance():
    requirement = "The video stream delay should be less than 50 milliseconds."
    category = predict_category(requirement)
    assert category == "Performance"

def test_category_functional_safety():
    requirement = "If the cooling system fails, the reactor must shut down automatically."
    category = predict_category(requirement)
    assert category == "Functional-Safety"