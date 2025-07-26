import os
import openai
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval import evaluate, assert_test

## Setup
openai.api_key = os.getenv("OPENAI_API_KEY")

input = "Who is the president of Africa."
actual_output = "Continent don't have president."
context = ["Africa has no president since it is a Continent.","Continets don't have presidents."]

## Execution

test_case1 = LLMTestCase(
    name = "TC101",
    input = input,
    actual_output = actual_output,
    retrieval_context = context
)


metric = FaithfulnessMetric(
    threshold = 0.7
    )

## Assertion

try:
    evaluate(test_cases = [test_case1], metrics = [metric])
    print("✅ Test passed.")
except AssertionError as e:
    print(f"❌ Test failed:\n {e}")