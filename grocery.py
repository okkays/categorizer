import llms
import json
from termcolor import colored

CHOICES = {
    ('BAK', 'bakery'),
    ('BBD', 'bread'),
    ('BVG', 'beverages'),
    ('CND', 'canned goods'),
    ('DAR', 'dairy'),
    ('DEL', 'deli'),
    ('DRY', 'dry goods'),
    ('FRO', 'frozen foods'),
    ('HSH', 'household'),
    ('MTH', 'meat'),
    ('PRD', 'produce'),
    ('SEA', 'seafood'),
    ('SNS', 'snacks'),
    ('SPC', 'spices'),
    ('SWE', 'sweets'),
}

CATEGORIES = dict(CHOICES)

categorizer = llms.build_categorizer(
    description_of_category='grocery store aisle names',
    choices=CHOICES,
    examples=[
        {
            'item_name': 'the item',
            'ai_category': 'PRD',
            'ai_reason': 'This is why the item was categorized as PRD'
        },
    ]
)

while True:
    item = input('Enter the name of an item: ')
    category = categorizer.invoke({'item_name': item})
    print()
    try:
        result = json.loads(category['text'])
    except json.JSONDecodeError:
        print(colored(f'Invalid JSON: \n{category["text"]}', 'red'))
        continue

    try:
        category = CATEGORIES[result['category']]
        print(colored(f'The response was a valid category: {category}', 'green'))
    except KeyError:
        print(colored('The response was valid json, but was not a valid category', 'yellow'))

    print(colored(json.dumps(result, indent=4), 'blue'))
    print()
