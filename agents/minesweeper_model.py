"""
MineRL — Model Loader

Loads a fine-tuned causal LM for Minesweeper inference.
Defaults to greedy decoding for reliable structured JSON output.
"""

import os
import time
from typing import Optional, List

# ROCm fix for Qwen2.5 sliding window attention on AMD GPUs
os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")

from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"


class MinesweeperAgent:
    """Wraps a causal LM for single-step Minesweeper action generation."""

    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs):
        """
        Args:
            model_name: HuggingFace model ID or local path to fine-tuned model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def generate_response(
        self,
        message: "str | List[str]",
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> tuple:
        """
        Generate a Minesweeper action for one or more board prompts.

        Args:
            message: A single prompt string or a list of prompts for batch inference.
            system_prompt: System instruction injected before the user message.
            **kwargs: Generation parameters:
                max_new_tokens (int): default 64
                do_sample (bool): default False (greedy)
                temperature (float): only used when do_sample=True
                top_p (float): only used when do_sample=True
                repetition_penalty (float): optional
                tgps_show (bool): if True, return token count and generation time

        Returns:
            (response, token_count, generation_time)
            token_count and generation_time are None unless tgps_show=True.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a Minesweeper AI. "
                'Output ONLY valid JSON: {"type":"reveal"|"flag","row":R,"col":C}'
            )

        if isinstance(message, str):
            message = [message]

        all_messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": msg}]
            for msg in message
        ]

        texts = [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in all_messages
        ]

        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        tgps_show = kwargs.get("tgps_show", False)
        if tgps_show:
            start_time = time.time()

        gen_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", 64),
            do_sample=kwargs.get("do_sample", False),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = kwargs.get("temperature", 1.0)
            gen_kwargs["top_p"] = kwargs.get("top_p", 1.0)
        if "repetition_penalty" in kwargs:
            gen_kwargs["repetition_penalty"] = kwargs["repetition_penalty"]

        generated_ids = self.model.generate(**model_inputs, **gen_kwargs)

        if tgps_show:
            generation_time = time.time() - start_time

        batch_outs = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        batch_outs = [o.strip() for o in batch_outs]

        if tgps_show:
            token_len = sum(
                len(generated_ids[i]) - model_inputs.input_ids.shape[1]
                for i in range(len(generated_ids))
            )
            return batch_outs[0] if len(batch_outs) == 1 else batch_outs, token_len, generation_time

        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path or HF ID")
    args = parser.parse_args()

    agent = MinesweeperAgent(model_name=args.model)
    test_prompt = (
        "MINESWEEPER 4x4 MINES:3 FLAGS:0 LEFT:3\n"
        "FRONTIER (numbered cells with hidden neighbors):\n"
        "R0C0=1 flags:0 hidden:[(1,0)(1,1)]\n"
        'Output ONLY: {"type":"reveal"|"flag","row":R,"col":C}'
    )
    response, tl, tm = agent.generate_response(test_prompt, tgps_show=True, max_new_tokens=64)
    print(f"Response: {response}")
    if tl and tm:
        print(f"Tokens: {tl}, Time: {tm:.2f}s, TGPS: {tl/tm:.1f}")
