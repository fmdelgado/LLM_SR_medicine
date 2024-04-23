from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import json
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

def is_json(data):
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False


def restructure_dict(output, key_renaming):
    output = {key_renaming.get(old_key, old_key): value for old_key, value in output.items()}

    new_dict = {}
    for checkpoint in output.keys():
        if 'checkpoint' in checkpoint:
            checkpoint_name = checkpoint.split('_')[1]  # Remove the "checkpoint_" prefix
            new_dict[checkpoint_name] = {
                'label': output['checkpoint_' + checkpoint_name],
                'reason': output['reason_' + checkpoint_name]
            }
    return new_dict


def check_inclusion_criteria(article_dict):
    """
    Check if an article meets the inclusion criteria.
    :param article_dict: dictionary containing analysis of an article.
    :return: True if the article should be included, False otherwise.
    """
    response_boolean_vector = [bool(values['label']) for values in output.values()]

    # Loop through each checkpoint in the article's dictionary
    for checkpoint, values in article_dict.items():
        # Check if the 'label' value for this checkpoint is 'False'
        if values['label'] == 'False':
            # If it is, return False (article should not be included)
            return False, response_boolean_vector

    # If all of the inclusion criteria were met, return True (article should be included)
    return True, response_boolean_vector


def generate_prompt_for_criteria_old(checkpoints_dict):
    response_schemas = []

    for i, (checkpoint, description) in enumerate(checkpoints_dict.items(), 1):
        response_schemas.append(
            ResponseSchema(
                name=f"checkpoint{i}",
                description=f"True/False, depending on whether {checkpoint} applies to the text."
            )
        )
        response_schemas.append(
            ResponseSchema(
                name=f"reason{i}",
                description=f"The reason for the decision made on {checkpoint}. {description}"
            )
        )

    # The parser that will look for the LLM output in my schema and return it back to me
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("""Given the following checkpoints with their description from the user, \
                                                       assess whether they apply or not to the text you receive as input below \
                                                       and provide a brief explanation on the why. \n
                                                        {format_instructions}\n{user_prompt}""")
        ],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
    )

    key_renaming = {}
    keys_list = list(checkpoints_dict.keys())
    for i in range(len(keys_list)):
        # print(i, keys_list[i])
        key_renaming.update({f"checkpoint{i + 1}": f"checkpoint_{keys_list[i]}"})
        key_renaming.update({f"reason{i + 1}": f"reason_{keys_list[i]}"})

    labels_from_user = "\n%CHECKPOINTS:\n\n" + '\n\n'.join(
        [f"{key} : {value}" for key, value in checkpoints_dict.items()])
    label_query = prompt.format_prompt(user_prompt=labels_from_user)
    return label_query


def generate_prompt_for_criteria(checkpoints_dict):
    # Define your desired data structure.
    response_schemas = []

    for i, (checkpoint, description) in enumerate(checkpoints_dict.items(), 1):
        response_schemas.append(
            ResponseSchema(
                name=f"checkpoint{i}",
                description=f"True/False, depending on whether {checkpoint} applies to the text.",
                type='boolean',
            )
        )
        response_schemas.append(
            ResponseSchema(
                name=f"reason{i}",
                description=f"The reason for the decision made on {checkpoint}. {description}",
                type='string',
            )
        )

    # The parser that will look for the LLM output in my schema and return it back to me
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template="""Given the following checkpoints with their description from the user, assess whether they apply or not to the text you receive as input below and provide a brief explanation on the why.\n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )

    key_renaming = {}
    keys_list = list(checkpoints_dict.keys())

    for i in range(len(keys_list)):
        # print(i, keys_list[i])
        key_renaming.update({f"checkpoint{i + 1}": f"checkpoint_{keys_list[i]}"})
        key_renaming.update({f"reason{i + 1}": f"reason_{keys_list[i]}"})

    labels_from_user = "\n%CHECKPOINTS:\n\n" + '\n\n'.join([f"{key} : {value}" for key, value in checkpoints_dict.items()])
    label_query = prompt.format_prompt(query=labels_from_user)

    return label_query, output_parser


def restructure_dict(output, checkpoints_dict):
    key_renaming = {}
    keys_list = list(checkpoints_dict.keys())
    for i in range(len(keys_list)):
        # print(i, keys_list[i])
        key_renaming.update({f"checkpoint{i + 1}": f"checkpoint_{keys_list[i]}"})
        key_renaming.update({f"reason{i + 1}": f"reason_{keys_list[i]}"})

    output = {key_renaming.get(old_key, old_key): value for old_key, value in output.items()}

    new_dict = {}
    for checkpoint in output.keys():
        if 'checkpoint' in checkpoint:
            checkpoint_name = checkpoint.split('_')[1]  # Remove the "checkpoint_" prefix
            new_dict[checkpoint_name] = {
                'label': output['checkpoint_'+checkpoint_name],
                'reason': output['reason_'+checkpoint_name]
            }
    return new_dict
