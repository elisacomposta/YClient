DEFAULT_SUSCEPTIBILITY = 0
TRAIT_SUSCEPTIBILITY = {
    # NE
    "sensitive/nervous": 0.9,  
    "resilient/confident": 0.1,
    # OE
    "inventive/curious": 0.2, 
    "consistent/cautious": 0.6,
    # CO
    "efficient/organized": 0.2,  
    "extravagant/careless": 0.6,
    # EX
    "outgoing/energetic": 0.5,
    "solitary/reserved": 0.5,
    # AG
    "friendly/compassionate": 0.5, 
    "critical/judgmental": 0.5,
}

API_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}

OPINION_TAG_SCORE = {
    "NEUTRAL": 0,
    "SUPPORTIVE": 0.5,
    "STRONGLY SUPPORTIVE": 1,
    "OPPOSED": -0.5,
    "STRONGLY OPPOSED": -1,
}

SCORE_TO_OPINION_TAG = {v: k for k, v in OPINION_TAG_SCORE.items()}

BIAS_INSTRUCTIONS = {
    "base_confirmation": "\n - Keep in mind that you are simulating a real person in this role-play. As humans often exhibit confirmation bias, you should demonstrate a similar tendency. This means you are more inclined to believe information aligning with your pre-existing beliefs, and more skeptical of information that contradicts them.",
    "strong_confirmation": "\n - Remember, you are role-playing as a real person. You have a strong confirmation bias. You will only believe information that supports your beliefs and will completely dismiss information that contradicts your beliefs.",
}

LOCALES = {
    "english": "en",
    "italian": "it",
    "italiano": "it",
}

TRANSLATIONS = {
    "en": {
        "toxicity_0": "neutral",
        "toxicity_1": "provocative",
        "toxicity_2": "sharp",
        "toxicity_3": "aggressive",
        "toxicity_4": "hostile",
        "follow": "follow",
        "unfollow": "unfollow",
    },
    "it": {
        "toxicity_0": "neutrale",
        "toxicity_1": "pacato",
        "toxicity_2": "critico",
        "toxicity_3": "polemico",
        "toxicity_4": "acceso",
        "follow": "seguire",
        "unfollow": "smettere di seguire",
    }
}

TOPIC_DESCRIPTION = {
    "reddito di cittadinanza": "a state subsidy for people living in poverty, designed to ensure a minimum standard of living and promote employment integration. Supporters believe reddito di cittadinanza is a necessary tool for social protection and inclusion; opponents are concerned about potential work disincentives and system abuses. The most radical want to abolish it, others aim to reform it. Use the exact name 'reddito di cittadinanza' in your output. Do NOT translate, abbreviate, modify it, or mention it with '@'.",
    "immigration": "debates focus on border control, bilateral agreements, and managing irregular migration. Supporters advocate for inclusive immigration policies, humanitarian protection and integration; opponents prioritize national security and strict border enforcement.",
    "nuclear energy": "debates focus on whether to include it in the energy mix. Supporters cite energy security; opponents stress risks, costs, and favor renewables.",
    "civil rights": "covers gender equality, LGBTQIA+ rights and family structure. Supporters support expanding protections for LGBTQIA+ individuals, gender equality, and inclusive definitions of family; opponents prioritize traditional family models and may reject changes to marriage, parenting, or gender roles."
}