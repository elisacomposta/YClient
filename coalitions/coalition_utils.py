import json
from y_client.constants import OPINION_TAG_SCORE

def get_coalition_opinion(coalition, coalition_file):
    data = json.load(open(coalition_file, "r", encoding="utf-8"))

    if coalition not in data:
        print(f"Coalition '{coalition}' not found in coalition file")
        return None

    return data[coalition]['topic_opinion']

def get_coalition_program_str(coalition, coalition_file):
    s = ''
    coalition_opinion = get_coalition_opinion(coalition, coalition_file)
    
    for topic, values in coalition_opinion.items():
        label = values.get("label", "NEUTRAL").upper()
        description = values.get("description", "")
        formatted_line = f" - {topic}: [{label}] {description}"
        s += formatted_line + "\n"

    return s.strip()


def get_topic_opinions_and_descriptions(topics):
    topic_score = {}
    topic_description = {}

    for topic, values in topics.items():
        topic_key = topic.lower()
        label = values.get("label", "NEUTRAL").upper()
        description = values.get("description", "")
        numeric_score = OPINION_TAG_SCORE.get(label, 0)

        topic_score[topic_key] = numeric_score
        topic_description[topic_key] = description

    return topic_score, topic_description

