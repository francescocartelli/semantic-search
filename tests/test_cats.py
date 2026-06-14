from pathlib import Path
import json
import pytest

from engine import Engine
from model import Model

TEST_DIR = Path(__file__).parent
MODEL_DIR = TEST_DIR.parent / 'local'


@pytest.fixture(scope="module")
def sample_data():
    """Fixture to load test data once for the whole test session."""
    data_path = TEST_DIR / "test_cats.json"
    with open(data_path, "r") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def initialized_engine(sample_data):
    """Fixture to initialize the model and engine once, speeding up tests."""
    model = Model("all-MiniLM-L6-v2", local_folder_path=MODEL_DIR)
    engine = Engine(model.encode)
    engine.add(sample_data["documents"])
    return engine


def test_semantic_search_results(initialized_engine, sample_data):
    """Test that the engine successfully returns valid results for all test queries."""
    for query in sample_data["queries"]:
        results = initialized_engine.search(query, top_k=10)

        assert isinstance(results, list), f"Expected list for query '{query}', got {type(results)}"
        
        assert len(results) <= 10, f"Expected maximum of 10 results, got {len(results)}"
        
        assert len(results) > 0, f"Search returned 0 results for query '{query}'"

        for result in results:
            assert "score" in result, "Each result must contain a 'score' key"
            assert "query" in result, "Each result must contain a 'query' key"
            assert isinstance(result["score"], (int, float)), "Score must be a numeric value"
            assert 0.0 <= result["score"] <= 1.0, f"Score {result['score']} is out of standard bounds"