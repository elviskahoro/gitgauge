from exa_py import Exa

from git_gauge.helper_exa import exa_api_call
from git_gauge.tokens import TOKENS

result = exa_api_call(
    "What is pricing for https://reflex.dev?",
    client=Exa(TOKENS["EXA_API_KEY"]),
)
print(result)
