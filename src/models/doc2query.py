from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class Doc2Query:
    r"""
    Doc2Query model, generate query based on the input document.
    """

    def __init__(
        self, model_name="doc2query/msmarco-vietnamese-mt5-base-v1", device="cuda"
    ) -> None:
        r"""
        Initialize an instance of the `doc2query/msmarco-vietnamese-mt5-base-v1` model by downloading it from Huggingface Hub. To know more about this model, follow this [link](https://huggingface.co/doc2query/msmarco-vietnamese-mt5-base-v1).

        Params:
            model_name (`str`):
                default: `doc2query/msmarco-vietnamese-mt5-base-v1`

                A string, the model id of a pretrained model hosted on huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.

            device (`str`):
                default: `cuda`

                A string, the id of the device which the model will run on, such as `cpu` or `cuda`, etc.

        Examples:

        ```python
        >>> from src import Doc2Query

        >>> # Initialize the Doc2Query model.
        >>> doc2query = Doc2Query(device = "cuda")

        >>> # Generate the query for a document.
        >>> text = "Robert Oppenheimer (22 tháng 4 năm 1904 – 18 tháng 2 năm 1967) là một nhà vật lý lý thuyết người Mỹ."
        >>> doc2query(text)
        "robert oppenheimer là ai"

        ```"""
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def encode(self, inputs: str):
        """
        Tokenize the input document with the predefined tokenizer.
        """
        return self.tokenizer.encode(inputs, return_tensors="pt").to(self.device)

    def decode(self, output):
        """
        De-tokenize the output from the text generation with the predefined tokenizer.
        """
        return [
            self.tokenizer.decode(output[i], skip_special_tokens=True)
            for i in range(len(output))
        ]

    def __call__(
        self,
        doc: str,
        strategy: str = "greedy",
        **kwargs,
    ):
        """
        Generate query based on the input document

        List options

        Params:
            doc (`str`):

                A string, representing the document user want to generate query from.

            strategy (`str`):
                default: `greedy`

                A string, representing the method which the model will use during generation. Can only be one of these options:

                - `greedy_search`
                - `contrastive_search`
                - `sample`
                - `multinomial_sample`
                - `beam_search`
                - `beam_sample`
                - `group_beam_search`

            **kwargs:

                Arguments passed to `generate` method of the Huggingface model.

        Examples:

        ```python
        >>> from src import Doc2Query

        >>> # Initialize the Doc2Query model.
        >>> doc2query = Doc2Query(device = "cuda")

        >>> # Generate the query for a document using greedy search(default).
        >>> text = "Robert Oppenheimer (22 tháng 4 năm 1904 – 18 tháng 2 năm 1967) là một nhà vật lý lý thuyết người Mỹ."
        >>> doc2query(text)
        ['robert oppenheimer là ai']

        >>> # Generate the query for a document using beam search.
        >>> text = "Robert Oppenheimer (22 tháng 4 năm 1904 – 18 tháng 2 năm 1967) là một nhà vật lý lý thuyết người Mỹ."
        >>> doc2query(text, "beam", num_return_sequences=3)
        ['robert oppenheimer là ai',
         'robert oppenheimer là ai?',
         'robert oppenheimer sinh ra khi nào']

        ```"""
        input_ids = self.encode(doc)
        config = {"max_length": 1024}

        match strategy:
            case "greedy_search":
                pass
            case "contrastive_search":
                config["penalty_alpha"] = 0.7
                config["top_k"] = 20
            case "sample":
                config["do_sample"] = True
                config["temperature"] = 0.7
                config["top_k"] = 20
                config["top_p"] = 0.2
            case "multinomial_sample":
                config["do_sample"] = True
                config["num_beams"] = 20
                config["temperature"] = 0.7
            case "beam_search":
                config["early_stopping"] = True
                config["num_beams"] = 20
                config["no_repeat_ngram_size"] = 2
            case "beam_sample":
                config["early_stopping"] = True
                config["do_sample"] = True
                config["num_beams"] = 20
                config["temperature"] = 0.7
            case "group_beam_search":
                config["early_stopping"] = True
                config["num_beams"] = 20
                config["num_beam_groups"] = 10
                config["diversity_penalty"] = 1.0
            case _:
                pass

        config = {**config, **kwargs}

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                **config,
            ).to(self.device)

        return self.decode(output)
