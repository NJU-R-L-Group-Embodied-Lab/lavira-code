from openai import OpenAI
from PIL import Image
from habitat import logger

class LaViRA_API:

    def __init__(self, la_api_key=None, la_base_url=None, la_model_name="gpt-4-vision-preview",
                 va_model_name=None, va_api_key=None, va_base_url=None):
        self.la_client = OpenAI(
            api_key=la_api_key,
            base_url=la_base_url
        )
        self.la_model_name = la_model_name

        if va_model_name:
            self.va_client = OpenAI(
                api_key=va_api_key,
                base_url=va_base_url
            )
            self.va_model_name = va_model_name
        else:
            self.va_client = None
            self.va_model_name = None

        # 统计信息
        self.reset_stats()

    def image_to_base64(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def generate(self, messages, images=None, max_new_tokens=1024, temperature=0.7, use_la=False, **kwargs):
        import time
        t = time.time()
        if use_la and self.la_client:
            client = self.la_client
            model_name = self.la_model_name
            stats_key = 'Language Action Model'
        else:
            client = self.va_client
            model_name = self.va_model_name
            stats_key = 'Vision Action Model'

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )

            self.stats[stats_key]['calls'] += 1
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"API Call usage - {response.usage}")
                self.stats[stats_key]['input_tokens'] += response.usage.prompt_tokens or 0
                self.stats[stats_key]['output_tokens'] += response.usage.completion_tokens or 0
                self.stats[stats_key]['total_tokens'] += response.usage.total_tokens or 0

                logger.info(f"{stats_key.upper()} model usage - Input: {response.usage.prompt_tokens}, "
                            f"Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")

            logger.info(f'Generating uses {time.time() - t} seconds.')
            print(self.stats)
            import sys
            sys.stdout.flush()
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"API Call error with {'secondary' if use_la else 'primary'} model ({model_name}): {e}")
            logger.info('Forcing retry..')
            import time
            time.sleep(30)
            return self.generate(messages, images, max_new_tokens, temperature, use_la, **kwargs)
            return "Error: Failed to get response from API"

    def get_usage_stats(self):
        return self.stats.copy()

    def print_usage_stats(self):
        total_calls = self.stats['Language Action Model']['calls'] + self.stats['Vision Action Model']['calls']
        total_tokens = self.stats['Language Action Model']['total_tokens'] + self.stats['Vision Action Model']['total_tokens']
        if self.la_client:
            logger.info("=== MODEL USAGE STATISTICS ===")
            logger.info(f"Language Action Model ({self.la_model_name}):")
            logger.info(f"  - Calls: {self.stats['Language Action Model']['calls']}")
            logger.info(f"  - Input tokens: {self.stats['Language Action Model']['input_tokens']:,}")
            logger.info(f"  - Output tokens: {self.stats['Language Action Model']['output_tokens']:,}")
            logger.info(f"  - Total tokens: {self.stats['Language Action Model']['total_tokens']:,}")

        if self.va_client:
            logger.info(f"Vision Action Model ({self.va_model_name}):")
            logger.info(f"  - Calls: {self.stats['Vision Action Model']['calls']}")
            logger.info(f"  - Input tokens: {self.stats['Vision Action Model']['input_tokens']:,}")
            logger.info(f"  - Output tokens: {self.stats['Vision Action Model']['output_tokens']:,}")
            logger.info(f"  - Total tokens: {self.stats['Vision Action Model']['total_tokens']:,}")

        logger.info(f"TOTAL:")
        logger.info(f"  - Total calls: {total_calls}")
        logger.info(f"  - Total tokens: {total_tokens:,}")
        logger.info("===============================")

        return {
            'la': self.stats['Language Action Model'].copy(),
            'va': self.stats['Vision Action Model'].copy(),
            'total_calls': total_calls,
            'total_tokens': total_tokens
        }

    def reset_stats(self):
        self.stats = {
            'Language Action Model': {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            },
            'Vision Action Model': {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }

    def eval(self):
        pass
