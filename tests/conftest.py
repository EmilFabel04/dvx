import pytest

from dvx.models import load_model

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_REV = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"


@pytest.fixture
def tiny():
    return load_model(TINY, TINY_REV, "cpu")
