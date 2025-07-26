# We start with the smallest testable components which makes up our unit test
# We will be using DeepEval for this and we will be testing the #Correctness Metric

import openai
import os
from deepeval import assert_test,evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

openai.api_key = os.getenv("OPENAI_API_KEY")

## Setup
input = "The dog chased the cat up the tree. Who went up the tree?"
actual_output="The cat"
expected_output="The cat went up a tree"


## For Correctness (Execution)
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the actual output is correct with regard to the expected output.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    strict_mode=False,
    threshold= 0.5
)
test_case1 = LLMTestCase(
  name= "TC001",
  input=input,
  actual_output=actual_output,
  expected_output=expected_output
)

## You can use this to show Metric data as well
#correctness_metric.measure(test_case1)
#print(correctness_metric.score)
#print(correctness_metric.reason)
#print(correctness_metric.is_successful())

## Assertion
try:
    assert_test(test_case1, metrics=[correctness_metric])
    print("✅ Test passed.")
except AssertionError as e:
    print(f"❌ Test failed:\n {e}")