# LLMTesting

This project provides example unit tests for evaluating Large Language Model (LLM) outputs using [DeepEval](https://github.com/confident-ai/deepeval). It demonstrates how to test LLM responses for correctness, faithfulness, hallucination, relevancy, and summarization.
## Getting Started

1. **Create a virtual environment**  
   ```sh
   python -m venv venv_deepeval

2. **Activate the virtual environment (Window, Linux/MacOs)** 
    ```sh
   venv_deepeval\Scripts\activate       # Windows
   source venv_deepeval/bin/activate    # macOS/Linux  

3. **Install Dependencies** 
    ```sh
    pip install deepeval openai

4. **Set your OpenAI API key** 
    ```sh
    set OPENAI_API_KEY=your-api-key-here  # Windows
    export OPENAI_API_KEY=your-api-key-here  # macOS/Linux

5. **Run Test using the deepeval command**
    ```sh
    deepeval test run test_AdventasSummarization.py
    deepeval test run test_AdventasFaithfulness.py
    deepeval test run test_AdventasHallucination.py
    deepeval test run test_AdventasRelevency.py
    deepeval test run test_AdventasCorrectness.py


