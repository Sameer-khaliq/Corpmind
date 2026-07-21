import pytest
from unittest.mock import patch, MagicMock
from corpmind.agents import enrichment as ea
from corpmind.schemas.enrichment import EnrichmentResolution
from corpmind.schemas.extraction import NormalizedProduct


def _mock_product():
    return NormalizedProduct(
        item_id="b_test_enrich",
        supplier_id="sup_enrich_01",
        source_row_index=88,
        title="Nike Air Max Alpha Trainer 5",
        category="casual-shoes",
        brand="Nike",
        material=None
    )


@patch("corpmind.agents.enrichment.web_search")
@patch("corpmind.agents.enrichment.ChatGroq")
def test_enrichment_react_loop_filled_grounded(mock_chat_groq, mock_web_search):
    mock_web_search.return_value = [
        {"url": "https://nike-shoes-test.com", "title": "Nike Materials", "content": "The Max Alpha 5 features a mesh upper."}
    ]

    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_chat_groq.return_value = mock_llm
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    turn_1_resp = MagicMock()
    turn_1_resp.tool_calls = [{"name": "search_web", "args": {"query": "Nike Air Max material"}, "id": "call_1"}]
    turn_1_resp.content = ""
    
    turn_2_resp = MagicMock()
    turn_2_resp.tool_calls = []
    turn_2_resp.content = """
    {
        "field_name": "material",
        "enriched_value": "mesh",
        "source_url": "https://nike-shoes-test.com",
        "resolution": "filled_grounded"
    }
    """

   
    mock_llm_with_tools.invoke.side_effect = [turn_1_resp, turn_2_resp]

    product = _mock_product()
    result = ea.enrich_field(product, "material")

    assert result.field_name == "material"
    assert result.enriched_value == "mesh"
    assert result.resolution == EnrichmentResolution.FILLED_GROUNDED
    assert result.source_url == "https://nike-shoes-test.com"