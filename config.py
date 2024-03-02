import os
import dotenv

dotenv.load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
USE_AWS = os.getenv('USE_AWS', 'False') == 'True'
PRINT_ALL_PROMPTS = os.getenv('PRINT_ALL_PROMPTS', 'False') == 'True'
