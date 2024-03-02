CATEGORIZER_SYSTEM_PROMPT = """You are a JSON API server responsible for returning categories for items. Your job today is to return what {description_of_category} items fall into:

Allowed choices:
{choices}

Your job is to choose ONE choice from above given an item name. Your output will be a JSON object in the following rigid format.

```json
{{{{
    'reason': 'A short explanation of why you chose this category.'
    'category': 'The category you chose',
}}}}
```

Future prompts will consist of item names, which you will return categories for. Do not reply with any acknowledgement of the task. Only reply with your category and reason formatted as above. You are unable to write code comments.
"""

CATEGORIZER_HUMAN_PROMPT = """{{'item_name': {item_name}}}"""

CATEGORIZER_AI_EXAMPLE_PROMPT = """
{{
    "reason": "{ai_reason}",
    "category": "{ai_category}"
}}
"""
