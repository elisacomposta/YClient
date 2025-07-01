import random
import json
import numpy as np
from YClient.coalitions.coalition_utils import get_coalition_opinion
import faker
import scipy.stats as stats
try:
    from y_client import Agent, PageAgent
    from y_client.utils import compute_susceptibility
except:
    from y_client.classes.base_agent import Agent
    from y_client.classes.page_agent import PageAgent

def generate_user(config, owner=None,
        joined_on=None,
        original_id=None,
        political_leaning=None,
        toxicity_post_avg=None,
        toxicity_post_var=None,
        toxicity_comment=None,
        n_posts=None,
        n_comments=None,
        coalition_file=None, 
        is_misinformation=False):
    """
    Generate a fake user
    :param config: configuration dictionary
    :param owner: owner of the user
    :param political_leaning: political leaning of the user
    :param toxicity_post_avg: average toxicity of the user posts
    :param toxicity_post_var: variance of toxicity of the user posts
    :param n_posts: number of posts written by the user
    :param n_comments: number of comments written by the user
    :param coalition_file: coalition file to use for generating the user
    :param is_misinformation: whether the user is a misinformation agent or not

    :return: Agent object
    """
    locales = json.load(open("config_files/nationality_locale.json"))
    try:
        nationality = random.sample(config["agents"]["nationalities"], 1)[0]
    except:
        nationality = "American"
    fake = faker.Faker(locales[nationality])

    # Age
    try:
        age_range = random.choices(config["agents"]["age"]["ranges"], weights=config["agents"]["age"]["probabilities"], k=1)[0].split("-")
        age = fake.random_int(min=int(age_range[0]), max=int(age_range[1]))
    except:
        age = fake.random_int(min=config["agents"]["age"]["min"], max=config["agents"]["age"]["max"])
    
    # Gender
    try:
        gender = random.choices(config["agents"]["gender"]["categories"], weights=config["agents"]["gender"]['probabilities'], k=1)[0]
    except:
        gender = random.sample(["male", "female"], 1)[0]

    if coalition_file is None:
        toxicity_post_avg = random.random()
        toxicity_comment = random.random()
        activity_post = random.random()
        activity_comment = random.random()
    else:
        all_data = json.load(open(coalition_file, "r", encoding="utf-8"))

        # Toxicity
        if toxicity_post_avg is None and toxicity_comment is None:
            try:
                toxicity_post_avg = sample_continuous_distribution(all_data[political_leaning]['toxicity']['post'])
                toxicity_comment = sample_continuous_distribution(all_data[political_leaning]['toxicity']['comment'])
            except:
                print("\nError in initializing toxicity from coalition file. Using random values")
                toxicity_post_avg = random.random()
                toxicity_comment = random.random()

        # Number of posts and comments
        if n_posts is None and n_comments is None:
            try:
                n_posts = sample_discrete_distribution(all_data[political_leaning]['content_count']['post'])
                n_comments = sample_discrete_distribution(all_data[political_leaning]['content_count']['comment'])
            except:
                print("\nError in initializing activity from coalition file. Using random values")
                n_posts = random.randint(0, 100)
                n_comments = random.randint(0, 100)

        # Activity
        n_post_max = all_data[political_leaning]['content_count']['post']['n_max']
        n_comment_max = all_data[political_leaning]['content_count']['comment']['n_max']

        activity_post = round(min(np.log1p(n_posts) / np.log1p(n_post_max), 1.0), 3)
        activity_comment = round(min(np.log1p(n_comments) / np.log1p(n_comment_max), 1.0), 3)

    first_name = fake.first_name_male() if gender == 'male' else fake.first_name_female()
    last_name = fake.last_name()
    name = f"{first_name} {last_name}"
    
    email = f"{name.replace(' ', '.')}@{fake.free_email_domain()}"
    pwd = fake.password()
    language = fake.random_element(elements=(config["agents"]["languages"]))
    ag_type = fake.random_element(elements=(config["agents"]["llm_agents"]))
    big_five = {
        "oe": fake.random_element(elements=(config["agents"]["big_five"]["oe"])),
        "co": fake.random_element(elements=(config["agents"]["big_five"]["co"])),
        "ex": fake.random_element(elements=(config["agents"]["big_five"]["ex"])),
        "ag": fake.random_element(elements=(config["agents"]["big_five"]["ag"])),
        "ne": fake.random_element(elements=(config["agents"]["big_five"]["ne"])),
    }
    if not is_misinformation:
        susceptibility = compute_susceptibility(list(big_five.values()))
    else:
        susceptibility = 0.05

    toxicity = fake.random_element(elements=(config["agents"]["toxicity_levels"]))
    education_level = fake.random_element(elements=(config["agents"]["education_levels"]))
    if political_leaning is None:
        political_leaning = fake.random_element(elements=(config["agents"]["political_leanings"]))

    interests = fake.random_elements(
        elements=list(config["agents"]["interests"]),
        length=fake.random_int(
            min=config["agents"]["n_interests"]["min"],
            max=config["agents"]["n_interests"]["max"],
        ),
        unique=True
    )

    try:
        round_actions = fake.random_int(
            min=config["agents"]["round_actions"]["min"],
            max=config["agents"]["round_actions"]["max"],
        )
    except:
        round_actions = 3

    api_key = config["servers"]["llm_api_key"]
    api_type = config["servers"]["llm_api_type"]

    agent = Agent(
        name=name.replace(" ", ""),
        pwd=pwd,
        email=email,
        age=age,
        ag_type=ag_type,
        interests=list(interests),
        config=config,
        big_five=big_five,
        language=language,
        education_level=education_level,
        owner=owner,
        round_actions=round_actions,
        gender=gender,
        nationality=nationality,
        toxicity=toxicity,
        api_key=api_key,
        is_page=0,
        api_type=api_type,
        joined_on=joined_on,
        original_id=original_id,
        leaning=political_leaning,
        toxicity_post_avg=toxicity_post_avg,
        toxicity_post_var=toxicity_post_var,
        toxicity_comment=toxicity_comment,
        activity_post=activity_post,
        activity_comment=activity_comment,
        coalition_program=get_coalition_opinion(political_leaning, coalition_file) if coalition_file else None,
        is_misinfo=1 if is_misinformation else 0,
        susceptibility=susceptibility,
    )

    if not hasattr(agent, "user_id"):
        print("Generated voter with no user_id")
        return None
    
    agent.init_opinions()
    return agent

def generate_page(config, owner=None, name=None, feed_url=None):
    """
    Generate a fake page
    :param config: configuration dictionary
    :param name: name of the page
    :param feed_url: feed url of the page
    :return: Agent object
    """

    fake = faker.Faker()

    try:
        round_actions = fake.random_int(
            min=config["agents"]["round_actions"]["min"],
            max=config["agents"]["round_actions"]["max"],
        )
    except:
        round_actions = 3

    big_five = {
        "oe": fake.random_element(elements=(config["agents"]["big_five"]["oe"])),
        "co": fake.random_element(elements=(config["agents"]["big_five"]["co"])),
        "ex": fake.random_element(elements=(config["agents"]["big_five"]["ex"])),
        "ag": fake.random_element(elements=(config["agents"]["big_five"]["ag"])),
        "ne": fake.random_element(elements=(config["agents"]["big_five"]["ne"])),
    }

    api_key = config["servers"]["llm_api_key"]

    email = f"{name.replace(' ', '.')}@{fake.free_email_domain()}"

    page = PageAgent(
        name=name,
        pwd="",
        email=email,
        age=0,
        ag_type=fake.random_element(elements=(config["agents"]["llm_agents"])),
        leaning=None,
        interests=[],
        config=config,
        big_five=big_five,
        language=None,
        education_level=None,
        owner=owner,
        round_actions=round_actions,
        gender=None,
        nationality=None,
        toxicity=None,
        api_key=api_key,
        feed_url=feed_url,
        is_page=1,
    )

    return page

def sample_continuous_distribution(entry):
    dist_name = entry['distribution']
    params = entry['params']
    dist = getattr(stats, dist_name)

    shape_keys = dist.shapes.split(', ') if dist.shapes else []
    shape_params = [params[k] for k in shape_keys]
    loc = params.get('loc', 0)
    scale = params.get('scale', 1)

    val = dist.rvs(*shape_params, loc=loc, scale=scale)
    return round(val, 3)

def sample_discrete_distribution(entry):
    zero_prob = entry.get('zero_prob', 0)
    if np.random.rand() < zero_prob:
        return 0

    dist_name = entry['distribution']
    params = entry['params']
    dist = getattr(stats, dist_name)

    if dist_name == 'poisson':
        mu = params['mu']
        val = dist.rvs(mu)
        while val == 0:
            val = dist.rvs(mu)
        return val

    elif dist_name == 'nbinom':
        n = params['n']
        p = params['p']
        val = dist.rvs(n, p)
        while val == 0:
            val = dist.rvs(n, p)
        return val

    else:
        raise ValueError(f"Unsupported discrete distribution: {dist_name}")
