from ai_agent_qbr.api_models import ChatRequest, OutputFinal


def test_chat_request_parses_metadata():
    request = ChatRequest.model_validate(
        {"message": "hello", "metadata": {"client_name": "Acme"}}
    )
    assert request.metadata is not None
    assert request.metadata.client_name == "Acme"


def test_output_final_type():
    payload = OutputFinal(data={"content": "done"})
    assert payload.status == "final"
