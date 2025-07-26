import openai
import os
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import assert_test,evaluate

openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the shared metric
metric = AnswerRelevancyMetric(threshold=0.7, include_reason=True)

# Define multiple test cases (Execution)
test_case1 = LLMTestCase(
        name = "TC301",
        input="What if these shoes don't fit?",
        actual_output="You have 30 days to get a full refund at no extra cost.",

    )

test_case2 = LLMTestCase(
        name = "TC302",
        input="What if these shoes don't fit?",
        actual_output="Sadza ngemuriwo",
    )


# Evaluate all test cases

try:
    evaluate(test_cases=[test_case1,test_case2], metrics=[metric])
    print("✅ Test passed.")
except AssertionError as e:
    print(f"❌ Test failed:\n {e}")