from typing import Any, List
import torch
import os
# os.environ["HF_HOME"] = "/nlp/cache/"
from transformers import pipeline

from .checker_base import CheckerBase

LABELS = ["Entailment", "Neutral", "Contradiction"]

Zephyr_CHECKING_PROMPT_Q = \
    """I have a claim made by a language model to a question, please help me for checking whether the claim can be entailed according to the provided reference which is related to the question. 
    The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

    If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
    If the claim is contradicted with the reference, answer 'Contradiction'.
    If the reference is not relevant to the claim or DOES NOT contain information to verify the claim, answer 'Neutral'. 

    Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

    ### Question:
    {question}

    ### Reference:
    {reference}

    ### Claim:
    {claim}

    Your answer should be only a single word in ['Entailment', 'Neutral', 'Contradiction']
    """

Zephyr_CHECKING_PROMPT = \
    """I have a claim made by a language model, please help me for checking whether the claim can be entailed according to the provided reference. 
    The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

    If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
    If the claim is contradicted with the reference, answer 'Contradiction'.
    If the reference is not relevant to the claim or DOES NOT contain information to verify the claim, answer 'Neutral'. 

    Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

    ### Reference:
    {reference}

    ### Claim:
    {claim}

    Your answer should be only a single word in ['Entailment', 'Neutral', 'Contradiction']
    """


class ZephyrChecker(CheckerBase):
    def __init__(
            self,
            device=0
    ):
        super().__init__()
        self.device = device
        os.environ["TRANSFORMERS_CACHE"] = "/nlp/cache/"
        self.model = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16,
                              device=device)
        self.prompt_temp = Zephyr_CHECKING_PROMPT
        self.prompt_temp_wq = Zephyr_CHECKING_PROMPT_Q

    @torch.no_grad()
    def _check(
            self,
            claims: List,
            references: List,
            response: str,
            question: str,
    ):
        ret_labels = []
        for claim, reference in zip(claims, references):
            if isinstance(claim, list):
                assert len(claim) == 3
                claim = f"({claim[0]}, {claim[1]}, {claim[2]})"
            if question is None:
                prompt = self.prompt_temp.format(
                    reference=reference,
                    claim=claim
                )
            else:
                prompt = self.prompt_temp_wq.format(
                    question=question,
                    reference=reference,
                    claim=claim
                )
            message = [{'role': 'user',
                        'content': prompt}]
            message = self.model.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

            response = self.model(message, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95,
                                  pad_token_id=self.model.tokenizer.eos_token_id)

            output = response[0]["generated_text"]
            cutline = "<|assistant|>"
            idx_output = output.find(cutline)
            refined_output = output[idx_output + len(cutline):]

            if refined_output and len(refined_output):
                label = None
                if self.label_contradiction.lower() in refined_output.lower():
                    label = self.label_contradiction
                elif self.label_entailment.lower() in refined_output.lower():
                    label = self.label_entailment
                else:
                    label = self.label_neutral
                ret_labels.append(label)
            else:
                raise 'None or empty string'

        return ret_labels
