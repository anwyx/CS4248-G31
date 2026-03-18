from meme_pipeline.grounding.factory import NoOpGrounder


def test_grounding_returns_valid_schema_on_failure():
    result = NoOpGrounder().ground("missing.jpg", "cat")
    assert result.query == "cat"
    assert result.status == "not_found"
    assert result.bbox_xyxy is None
