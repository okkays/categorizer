import boto3

MODELS = [
    "anthropic.claude-v2",
    "anthropic.claude-v2:1",
    "meta.llama2-13b-chat-v1",
    "meta.llama2-70b-chat-v1",
    "amazon.titan-text-express-v1",
    "amazon.titan-text-lite-v1"
]


def get_bedrock_client(aws_access_key, aws_secret_key):
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    return bedrock_client


def get_examples(agent_client, kb_id, input, num_results):
    # TODO: error handling
    return agent_client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={
            'text': input[:1000]  # max length of AWS API
        },
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': num_results
            }
        },
    )['retrievalResults']
