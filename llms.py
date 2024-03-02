from typing import List, Tuple, Dict, Any
import aws
import config
import prompts

from langchain_community.chat_models import ChatOllama, BedrockChat


from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain.globals import set_verbose

import requests

set_verbose(config.PRINT_ALL_PROMPTS)

OLLAMA_URL = "http://localhost:11434"


def get_models():
    if config.USE_AWS:
        return aws.MODELS

    available_models = requests.get(f'{OLLAMA_URL}/api/tags')
    return [
        model.get('name', '')
        for model in
        available_models.json().get('models', [])
    ]


OFFLINE_MODEL_PARAMS = {
    'categorizer': {
        'model': 'neural-chat',
        'format': 'json',
        'temperature': 0.2,
        'repeat_penalty': 1.5,
        'num_predict': 128,
        'top_k': 5,
        'top_p': 0.1,
    },
}

AWS_MODEL_PARAMS = {
    'categorizer': {
        'model_id': aws.MODELS[0],
        'temperature': 0.2,
        'top_k': 1,
        'top_p': 1,
    },
}

AWS_KWARGS = {
    'model_id': 'model_id',
    'max_tokens_to_sample': 'max_tokens_to_sample',
}

OFFLINE_KWARGS = {
    'model_id': 'model',
    'max_tokens_to_sample': 'max_tokens_to_sample',
}


def translate_model_kwargs(kwargs, *, use_aws: bool):
    rosetta = AWS_KWARGS if use_aws else OFFLINE_KWARGS
    return {rosetta[k]: v for k, v in kwargs.items()}


def get_llm(purpose: str,
            *,
            extra_model_kwargs: Dict[str, Any] = None,
            aws_kwargs: Dict[str, Any] = None,
            use_aws: bool = config.USE_AWS,
            ) -> LLMChain:
    if not extra_model_kwargs:
        extra_model_kwargs = {}
    extra_model_kwargs = translate_model_kwargs(extra_model_kwargs,
                                                use_aws=use_aws)

    if not use_aws:
        return ChatOllama(base_url=OLLAMA_URL,
                          **OFFLINE_MODEL_PARAMS[purpose],
                          **extra_model_kwargs)

    if not aws_kwargs:
        aws_kwargs = {
            'aws_access_key': config.AWS_ACCESS_KEY,
            'aws_secret_key': config.AWS_SECRET_KEY,
        }
    bedrock = aws.get_bedrock_client(**aws_kwargs)
    aws_params = dict(AWS_MODEL_PARAMS[purpose])
    model_id = extra_model_kwargs.pop('model_id', aws_params.pop('model_id', ''))
    return BedrockChat(
        client=bedrock,
        model_id=model_id,
        model_kwargs={
            **aws_params,
            **extra_model_kwargs,
        }
    )


def build_categorizer(*,
                      description_of_category: str,
                      choices: List[Tuple[str, str]],
                      aws_kwargs: Dict[str, Any] = None,
                      examples,
                      ) -> LLMChain:
    llm = get_llm('categorizer', aws_kwargs=aws_kwargs)

    formatted_choices = '\n'.join([f'- {name} ({description})' for name, description in choices])

    example_prompt = ChatPromptTemplate(
        input_variables=['item_name', 'ai_category', 'ai_reason'],
        messages=[
            HumanMessagePromptTemplate.from_template(prompts.CATEGORIZER_HUMAN_PROMPT),
            AIMessagePromptTemplate.from_template(prompts.CATEGORIZER_AI_EXAMPLE_PROMPT),
        ]
    )

    prompt = ChatPromptTemplate(
        input_variables=['item_name'],
        messages=[
            SystemMessagePromptTemplate.from_template(
                prompts.CATEGORIZER_SYSTEM_PROMPT.format(
                    description_of_category=description_of_category,
                    choices=formatted_choices
                )
            ),
            FewShotChatMessagePromptTemplate(example_prompt=example_prompt,
                                             examples=examples),
            HumanMessagePromptTemplate.from_template(prompts.CATEGORIZER_HUMAN_PROMPT),
        ]
    )

    return LLMChain(llm=llm, prompt=prompt)
