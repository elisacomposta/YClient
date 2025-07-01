import os
import numpy as np
import requests
from autogen import AssistantAgent

def query_huggingface(prompt: str, llm_config: object) -> str:
    """
    Query the Hugging Face API with the given prompt and configuration.

    :param prompt: The prompt to query the API with.
    :param llm_config: The configuration object.
    :return: The generated text from the API response.
    """
    # model URL
    base_url = llm_config['config_list'][0]['base_url']
    api_model = llm_config['config_list'][0]['model']
    hf_api_url = f"{base_url}/{api_model}"

    # API key
    api_key_config = llm_config['config_list'][0]['api_key']
    hf_api_key = api_key_config if api_key_config != "NULL" else os.getenv("HF_API_KEY_2")

    # API request
    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "temperature": llm_config['temperature'],  
            "top_p": 0.9,        
            "top_k": 50,
            "max_new_tokens": llm_config['max_tokens'],
            "seed": np.random.randint(0, 100000)
        }
    }
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    response = requests.post(hf_api_url, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Errore: {response.status_code} - {response.text}"
    
    result = response.json()
    text = result[0].get("generated_text", "").replace(prompt, "").strip()
    return text
    

def query_assistant(name, llm_config, agent_prompt, handler_prompt, handler_message) -> str:
    """
    Query the AssistantAgent with the given prompts and configuration.
    
    :param name: The name of the AssistantAgent.
    :param llm_config: The configuration object.
    :param agent_prompt: The prompt for the AssistantAgent.
    :param handler_prompt: The prompt for the Handler.
    :param handler_message: The message for the Handler to send to the agent.
    :return: The generated chat messages
    """
    llm_config["seed"] = np.random.randint(0, 100000)

    u1 = AssistantAgent(
        name=name,
        llm_config=llm_config,
        system_message=agent_prompt,
        max_consecutive_auto_reply=1,
    )

    u2 = AssistantAgent(
        name=f"Handler",
        llm_config=llm_config,
        system_message=handler_prompt,
        max_consecutive_auto_reply=1,
    )

    u2.initiate_chat(
        u1,
        message=handler_message,
        silent=True,
        max_round=1,
    )

    chat_messages_u1 = u1.chat_messages[u2]
    chat_messages_u2 = u2.chat_messages[u1]

    u1.reset()
    u2.reset()

    return chat_messages_u1, chat_messages_u2

def query_model(prompt, llm_config) -> str:
    model_name = llm_config['config_list'][0]['model']
    base_url = llm_config['config_list'][0]['base_url']
    endpoint = f"{base_url}/v1/chat/completions"

    data = {
        "model": f"{model_name}.gguf",
        "temperature": llm_config['temperature'],
        "messages": [{"role": "user", "content": prompt}]
    }

    res = requests.post(endpoint, json=data)
    if res.status_code != 200:
        raise Exception(f"Model query error: {res.status_code} - {res.text}")

    result = res.json()
    text = result['choices'][0]['message']['content'].strip()
    return text
    