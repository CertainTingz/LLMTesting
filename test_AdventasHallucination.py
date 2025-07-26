# We start with the smallest testable components which makes up our unit test
# We will be using DeepEval for this and we will be testing the #Summarization Metric

import openai
import os
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate, assert_test

openai.api_key = os.getenv("OPENAI_API_KEY")


## Hallucination Test
## SetUp

input_text= "What is the population of Boravia"
actual_output = """ It seems you are asking about the population of a place called Boravia.
 However, "Boravia" doesn't appear to be a recognized location in public records. 
 Could you please double-check the name or provide more context?
If you're referring to a fictional or lesser-known area, 
let me know and I might still find something interesting to share!"""
context = [ "Boravia does not exist",
           "In the comics, Boravia is a fictional kingdom in Europe that faced a civil war"

]

#Creating our Test Case (Execution)
test_case1= LLMTestCase( 
    name = "TC201",
    input=input_text, 
    actual_output=actual_output,
    context = context)


## Now running the DeepEval Hallucination Metric 
metric = HallucinationMetric(threshold=0.5)
metric.measure(test_case1)

## Assertion
print(metric.score)
print(metric.reason)
print(metric.is_successful())


try:
    evaluate (test_cases= [test_case1], metrics=[metric])
    print("✅ Test passed.")
except AssertionError as e:
    print(f"❌ Test failed:\n {e}")

