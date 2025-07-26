# We start with the smallest testable components which makes up our unit test
# We will be using DeepEval for this and we will be testing the #Summarization Metric

import openai
import os
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test

openai.api_key = os.getenv("OPENAI_API_KEY")


## Text Summarization
## Setup

original_text="""XAI hopes to help users of AI-powered systems perform more 
effectively by improving their understanding of how those systems reason.
XAI may be an implementation of the social right to explanation.
"""

summary="""Explainable AI (XAI) aims to enhance user performance and trust by
 clarifying how AI systems make decisions. It can act as a mechanism to fulfill 
 the social right to explanation, even if not legally required. XAI emphasizes 
 making AI actions—past, present, and future—transparent and based on identifiable
 information. This transparency enables users to confirm existing knowledge, 
 question assumptions, and develop new insights.
"""

## Creating our Test Case (Execution)
test_case1 = LLMTestCase(
    name= "TC401",
    input=original_text,
    actual_output=summary)
metric = SummarizationMetric(threshold=0.7)

# Now running the DeepEval Summarization Metric 

metric.measure(test_case1)
print(metric.score)
print(metric.reason)
print(metric.is_successful())

try:
    assert_test(test_case1, metrics=[metric])
    print("✅ Test passed.")
except AssertionError as e:
    print(f"❌ Test failed:\n {e}")
