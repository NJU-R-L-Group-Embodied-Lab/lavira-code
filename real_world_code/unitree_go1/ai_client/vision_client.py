"""
OpenAI-compatible LaViRA API client.

Maintains two independently configurable endpoints, mirroring the
``LaViRA_API`` class in the simulation repo
(https://github.com/NJU-R-L-Group-Embodied-Lab/lavira-code):

- Language Action (LA) model: used for direction decisions. The paper
  recommends a language-strong VLM such as ``gemini-3.1-pro`` (API) or
  ``Qwen3.5-27B-Q4`` (local).
- Vision Action (VA) model: used for target detection / bounding-box outputs.
  The paper recommends ``Qwen3.5-27B`` (API) or ``Qwen3.5-27B-Q4`` (local).
"""
import sys
import time
import traceback

from openai import OpenAI


class LaViRAVisionClient:
    """Two-endpoint OpenAI-compatible client used by the LaViRA real-world stack."""

    def __init__(self, la_api_key=None, la_base_url=None, la_model_name="gpt-4o",
                 va_api_key=None, va_base_url=None, va_model_name=None):
        self.la_client = OpenAI(api_key=la_api_key, base_url=la_base_url)
        self.la_model_name = la_model_name

        if va_model_name:
            self.va_client = OpenAI(
                api_key=va_api_key or la_api_key,
                base_url=va_base_url or la_base_url,
            )
            self.va_model_name = va_model_name
        else:
            self.va_client = None
            self.va_model_name = None

        self.stats = {
            "Language Action Model": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "Vision Action Model":   {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    # ------------------------------------------------------------------ #
    # Core generate
    # ------------------------------------------------------------------ #
    def generate(self, messages, max_new_tokens=1024, temperature=0.0, use_la=False):
        """Send a multimodal chat completion and return the assistant content string.

        ``use_la=True`` routes the call to the Language Action model; otherwise
        the call is routed to the Vision Action model.
        """
        t = time.time()
        if use_la:
            client, model_name, stats_key = self.la_client, self.la_model_name, "Language Action Model"
        else:
            if self.va_client is None:
                client, model_name, stats_key = self.la_client, self.la_model_name, "Language Action Model"
            else:
                client, model_name, stats_key = self.va_client, self.va_model_name, "Vision Action Model"

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

            self.stats[stats_key]["calls"] += 1
            usage = getattr(response, "usage", None)
            if usage:
                self.stats[stats_key]["input_tokens"] += usage.prompt_tokens or 0
                self.stats[stats_key]["output_tokens"] += usage.completion_tokens or 0
                self.stats[stats_key]["total_tokens"] += usage.total_tokens or 0

            duration = time.time() - t
            prompt_speed = (usage.prompt_tokens / duration) if usage and usage.prompt_tokens else 0.0
            output_speed = (usage.completion_tokens / duration) if usage and usage.completion_tokens else 0.0

            content = response.choices[0].message.content
            return content, {
                "duration": duration,
                "prompt_speed": prompt_speed,
                "output_speed": output_speed,
                "usage": usage,
            }

        except Exception as e:
            print(f"API call error with {stats_key} ({model_name}): {e}")
            traceback.print_exc()
            sys.stdout.flush()
            time.sleep(10)
            return self.generate(messages, max_new_tokens, temperature, use_la)

    def generate_with_la(self, messages, max_new_tokens=1024, temperature=0.0):
        """Convenience wrapper that always uses the Language Action model."""
        return self.generate(messages, max_new_tokens, temperature, use_la=True)

    def generate_with_va(self, messages, max_new_tokens=1024, temperature=0.0):
        """Convenience wrapper that always uses the Vision Action model."""
        return self.generate(messages, max_new_tokens, temperature, use_la=False)

    # ------------------------------------------------------------------ #
    # Usage statistics
    # ------------------------------------------------------------------ #
    def get_model_info(self):
        return {
            "la_model": self.la_model_name,
            "va_model": self.va_model_name if self.va_client else None,
            "has_va": self.va_client is not None,
        }

    def print_usage_stats(self):
        la = self.stats["Language Action Model"]
        va = self.stats["Vision Action Model"]
        total_calls = la["calls"] + va["calls"]
        total_tokens = la["total_tokens"] + va["total_tokens"]

        print("=== MODEL USAGE STATISTICS ===")
        print(f"Language Action ({self.la_model_name}):")
        print(f"  - Calls: {la['calls']}  Tokens: {la['total_tokens']:,}")
        if self.va_client:
            print(f"Vision Action ({self.va_model_name}):")
            print(f"  - Calls: {va['calls']}  Tokens: {va['total_tokens']:,}")
        print(f"TOTAL: {total_calls} calls, {total_tokens:,} tokens")
        print("==============================")

        return {"la": la.copy(), "va": va.copy(),
                "total_calls": total_calls, "total_tokens": total_tokens}

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
