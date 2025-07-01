from .constants import TRANSLATIONS, LOCALES, DEFAULT_SUSCEPTIBILITY, TRAIT_SUSCEPTIBILITY, OPINION_TAG_SCORE
import numpy as np
import re

def toxicity_to_label(toxicity: float, language: str = "english") -> str:  
    """
    Converts a toxicity score to a label, using a predefined mapping of toxicity levels to labels.
    """
    if toxicity is None or toxicity < 0.2:
        label =  "toxicity_0"
    elif toxicity < 0.4:
        label =  "toxicity_1"
    elif toxicity < 0.6:
        label =  "toxicity_2"
    elif toxicity < 0.8:
        label =  "toxicity_3"
    else:
        label =  "toxicity_4"
    
    locale = LOCALES.get(language.lower())
    return TRANSLATIONS[locale].get(label, label)


def compute_susceptibility(traits: list) -> float:
    """
    Computes the mean of the susceptibility scores for the given traits.
    """
    try:
        lambdas = [TRAIT_SUSCEPTIBILITY[trait] for trait in traits]
        return round(np.mean(lambdas), 3)

    except:
        print(f"Error: One or more personality traits not found in susceptibility scores. Assigning default value.")
        return DEFAULT_SUSCEPTIBILITY

def parse_opinions(response_text, topics):
    """
    Parses the response text to extract opinions on specified topics.
    
    Each line in the response text is expected to be in the format:
    "topic: [label] thought"
    where 'label' is one of the keys in OPINION_TAG_SCORE.
    """
    results = []
    pattern = r'^(.*?)\:\s+\[(.*?)\]\s+(.*)$'

    for line in response_text.splitlines():
        match = re.match(pattern, line)
        if match:
            raw_topic = match.group(1).strip().lower()
            label = match.group(2).strip()
            thought = match.group(3).strip()

            clean_topic = re.sub(r'[^a-zA-Z\s]', '', raw_topic).strip()
            if clean_topic in topics and label in OPINION_TAG_SCORE:
                results.append({
                    'topic': clean_topic,
                    'score': OPINION_TAG_SCORE[label],
                    'description': thought
                })

    return results

def opinions_to_str(opinions, topics):
    current_opinions = ""

    for topic in topics:
        if topic in opinions:
            current_opinions += f" - {topic.capitalize()}: [{opinions[topic]['label']}] {opinions[topic]['description']}\n"
    
    return current_opinions