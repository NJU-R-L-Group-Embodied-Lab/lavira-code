"""
OpenAI-compatible LaViRA API client.

The client maintains two independently configurable endpoints, mirroring the
``LaViRA_API`` class in the simulation repo
(https://github.com/NJU-R-L-Group-Embodied-Lab/lavira-code):

- Language Action (LA) model: used for direction decisions. The paper
  recommends a language-strong VLM such as ``gpt-4o`` or ``gemini-2.5-pro``.
- Vision Action (VA) model: used for target detection / bounding-box outputs.
  The paper recommends ``Qwen2.5-VL-32B-Instruct``.
"""
import base64
import io
import sys
import time
import traceback
import numpy as np
from PIL import Image
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
            'Language Action Model': {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
            'Vision Action Model': {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
        }

    def image_to_base64(self, image):
        """Encode a PIL.Image or numpy array as base64 JPEG."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def generate(self, messages, images=None, max_new_tokens=1024, temperature=0.7,
                 use_la=False, **kwargs):
        """Send a multimodal chat request and return the assistant content string.

        ``use_la=True`` routes the call to the Language Action model; otherwise
        the call is routed to the Vision Action model.
        """
        t = time.time()
        if use_la:
            client, model_name, stats_key = self.la_client, self.la_model_name, 'Language Action Model'
        else:
            if self.va_client is None:
                print("Warning: Vision Action model not configured, falling back to Language Action model.")
                client, model_name, stats_key = self.la_client, self.la_model_name, 'Language Action Model'
            else:
                client, model_name, stats_key = self.va_client, self.va_model_name, 'Vision Action Model'

        final_messages = []
        if messages and "content" in messages[0]:
            final_messages.append({"role": "user", "content": messages[0]["content"]})

        if not final_messages:
            print("Error: Messages format is incorrect or empty.")
            return "Error: Invalid message format."

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=final_messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

            self.stats[stats_key]['calls'] += 1
            if hasattr(response, 'usage') and response.usage:
                print(f"API call usage - {response.usage}")
                self.stats[stats_key]['input_tokens'] += response.usage.prompt_tokens or 0
                self.stats[stats_key]['output_tokens'] += response.usage.completion_tokens or 0
                self.stats[stats_key]['total_tokens'] += response.usage.total_tokens or 0

            print(f"Generation took {time.time() - t:.2f}s.")
            sys.stdout.flush()
            return response.choices[0].message.content

        except Exception as e:
            print(f"API call error with {stats_key} ({model_name}): {e}")
            print("Forcing retry...")
            traceback.print_exc()
            sys.stdout.flush()
            time.sleep(10)
            return self.generate(messages, images, max_new_tokens, temperature, use_la, **kwargs)

    def generate_with_la(self, messages, images=None, max_new_tokens=1024, temperature=0.7, **kwargs):
        """Convenience wrapper that forces the Language Action model."""
        return self.generate(messages, images, max_new_tokens, temperature, use_la=True, **kwargs)

    def generate_with_va(self, messages, images=None, max_new_tokens=1024, temperature=0.7, **kwargs):
        """Convenience wrapper that forces the Vision Action model."""
        return self.generate(messages, images, max_new_tokens, temperature, use_la=False, **kwargs)

    def get_model_info(self):
        """Return information about configured models."""
        return {
            "la_model": self.la_model_name,
            "va_model": self.va_model_name if self.va_client else None,
            "has_va": self.va_client is not None,
        }

    def get_usage_stats(self):
        """Return a shallow copy of the usage statistics dictionary."""
        return self.stats.copy()

    def print_usage_stats(self):
        """Print a detailed usage summary."""
        la_stats = self.stats['Language Action Model']
        va_stats = self.stats['Vision Action Model']
        total_calls = la_stats['calls'] + va_stats['calls']
        total_tokens = la_stats['total_tokens'] + va_stats['total_tokens']

        print("=== MODEL USAGE STATISTICS ===")
        print(f"Language Action Model ({self.la_model_name}):")
        print(f"  - Calls: {la_stats['calls']}")
        print(f"  - Input tokens: {la_stats['input_tokens']:,}")
        print(f"  - Output tokens: {la_stats['output_tokens']:,}")
        print(f"  - Total tokens: {la_stats['total_tokens']:,}")

        if self.va_client:
            print(f"Vision Action Model ({self.va_model_name}):")
            print(f"  - Calls: {va_stats['calls']}")
            print(f"  - Input tokens: {va_stats['input_tokens']:,}")
            print(f"  - Output tokens: {va_stats['output_tokens']:,}")
            print(f"  - Total tokens: {va_stats['total_tokens']:,}")

        print("TOTAL:")
        print(f"  - Total calls: {total_calls}")
        print(f"  - Total tokens: {total_tokens:,}")
        print("==============================")

        return {
            'la': la_stats.copy(),
            'va': va_stats.copy(),
            'total_calls': total_calls,
            'total_tokens': total_tokens,
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.stats = {
            'Language Action Model': {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
            'Vision Action Model': {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
        }

    def eval(self):
        """Compatibility shim for code that expects a torch-like ``eval`` method."""
        pass
