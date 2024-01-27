# FPTU Chatbot - Doc2Query

## In this README

- [Initial setup](#initial-setup)
- [Usage](#usage)

## Initial setup

1. Clone this repository to your machine.

   ```
   git clone https://github.com/rusano1612/fptu-chatbot-doc2query
   cd fptu-chatbot-doc2query
   ```

2. Create a Python 3.10 or newer virtual environment.

   ```
   conda create -n doc2query python=3.10
   conda activate doc2query
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage (Python)

1. Import the package.

   ```python
   from src import Doc2Query
   ```

2. Initialize the model.

   ```python
   doc2query = Doc2Query()
   ```

3. Doc2Query

   ```python
   text = "Robert Oppenheimer (22 tháng 4 năm 1904 – 18 tháng 2 năm 1967) là một nhà vật lý lý thuyết người Mỹ."
   doc2query(text, "greedy", num_return_sequences=1)
   ```
