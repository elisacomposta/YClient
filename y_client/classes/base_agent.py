from y_client.recsys.ContentRecSys import ContentRecSys
from y_client.recsys.FollowRecSys import FollowRecSys
from y_client.news_feeds.client_modals import Websites, Images, session, Agent_Custom_Prompt
from y_client.classes.annotator import Annotator
from sqlalchemy.sql.expression import func
from y_client.news_feeds.feed_reader import NewsFeed
from y_client.classes.time import SimulationSlot
from y_client.utils import opinions_to_str, parse_opinions, toxicity_to_label
from y_client.llm_helper import query_assistant, query_huggingface, query_model
import random
import requests
import json
import numpy as np
import re
from y_client.constants import API_HEADERS, BIAS_INSTRUCTIONS, LOCALES, SCORE_TO_OPINION_TAG, TRANSLATIONS, TOPIC_DESCRIPTION

__all__ = ["Agent", "Agents"]


class Agent(object):
    def __init__(
        self,
        name: str,
        email: str,
        pwd: str = None,
        age: int = None,
        interests: list = None,
        leaning: str = None,
        ag_type="llama3",
        load: bool = False,
        recsys: ContentRecSys = None,
        frecsys: FollowRecSys = None,
        config: dict = None,
        big_five: dict = None,
        language: str = None,
        owner: str = None,
        education_level: str = None,
        joined_on: int = None,
        round_actions: int = 3,
        gender: str = None,
        nationality: str = None,
        toxicity: str = "no",
        api_key: str = "NULL",
        api_type: str = "open_ai",
        is_page: int = 0,
        original_id: int = None,
        toxicity_post_avg: float = 0,
        toxicity_post_var: float = 0,
        toxicity_comment: float = 0,
        activity_post: float = 0,
        activity_comment: float = 0,
        coalition_program: dict = None,
        is_misinfo: int = False,
        susceptibility: float = 0,
        *args,
        **kwargs,
    ):
        """
        Initialize the Agent object.

        :param name: the name of the agent
        :param email: the email of the agent
        :param pwd: the password of the agent
        :param age: the age of the agent
        :param interests: the interests of the agent
        :param leaning: the leaning of the agent
        :param ag_type: the type of the agent
        :param load: whether to load the agent from file or not
        :param recsys: the content recommendation system
        :param frecsys: the follow recommendation system
        :param config: the configuration dictionary
        :param big_five: the big five personality traits
        :param language: the language of the agent
        :param owner: the owner of the agent
        :param education_level: the education level of the agent
        :param joined_on: the joined on date of the agent
        :param round_actions: the number of daily actions
        :param gender: the agent gender
        :param nationality: the agent nationality
        :param toxicity: the toxicity level of the agent, default is "no"
        :param api_key: the LLM server api key, default is NULL (self-hosted)
        :param is_page: whether the agent is a page or not, default is 0
        :param toxicity_post_avg: the average toxicity of the posts
        :param toxicity_post_var: the variance of the toxicity of the posts
        """
        self.session = requests.Session()

        if "web" in kwargs:

            self.__web_init(name=name, email=email,pwd=pwd, interests=interests, leaning=leaning,
                            ag_type=ag_type, load=load, recsys=recsys, age=age,
                            frecsys=frecsys, config=config, big_five=big_five, language=language, owner=owner, education_level=education_level,
                            joined_on=joined_on, round_actions=round_actions, gender=gender, nationality=nationality, toxicity=toxicity,
                            api_key=api_key, is_page=is_page, *args, **kwargs)
        else:
            self.emotions = config["posts"]["emotions"]
            self.actions_likelihood = config["simulation"]["actions_likelihood"]
            self.base_url = config["servers"]["api"]
            self.llm_base = config["servers"]["llm"]
            self.content_rec_sys_name = None
            self.follow_rec_sys_name = None
            self.opinion_model = config["simulation"]["opinion_model"]
            self.name = name
            self.email = email
            self.attention_window = int(config["agents"]["attention_window"])
            self.llm_v_config = {
                "url": config["servers"]["llm_v"],
                "api_key": config["servers"]["llm_v_api_key"] if (config["servers"]["llm_v_api_key"] is not None and config["servers"]["llm_v_api_key"] != "") else "NULL",
                "model": config["agents"]["llm_v_agent"],
                "temperature": config["servers"]["llm_v_temperature"],
                "max_tokens": config["servers"]["llm_v_max_tokens"]
            }
            self.is_page = is_page
            self.llm_language = config["agents"]["llm_language"]
            self.memory = ""

            if not load:
                self.language = language
                self.type = ag_type
                self.age = age
                self.interests = interests
                self.leaning = leaning
                self.pwd = pwd
                self.oe = big_five["oe"]
                self.co = big_five["co"]
                self.ex = big_five["ex"]
                self.ag = big_five["ag"]
                self.ne = big_five["ne"]
                self.owner = owner
                self.education_level = education_level
                sc = SimulationSlot(config)
                sc.get_current_slot()
                self.joined_on = sc.id if joined_on is None else joined_on
                self.round_actions = round_actions
                self.gender = gender
                self.nationality = nationality
                self.toxicity = toxicity
                self.original_id = original_id
                self.toxicity_posts_avg = toxicity_post_avg
                self.toxicity_posts_var = toxicity_post_var
                self.toxicity_comment = toxicity_comment
                self.activity_post = activity_post
                self.activity_comment = activity_comment
                self.coalition_program = coalition_program
                self.is_misinfo = is_misinfo
                self.susceptibility = susceptibility

                uid = self.__register()
                if uid is None:
                    pass
                else:
                    self.user_id = uid

            else:
                us = json.loads(self.__get_user())
                self.user_id = us["id"]
                self.type = us["user_type"]
                self.age = us["age"]

                if us["is_page"] == 0:
                    n_interests = random.randint(config["agents"]["n_interests"]["min"],
                                                    config["agents"]["n_interests"]["max"])
                    self.interests = n_interests
                    self.interests = self._get_interests(-1, n_max=n_interests)[0]
                else:
                    self.interests = []

                self.leaning = us["leaning"]
                self.pwd = us["password"]
                self.oe = us["oe"]
                self.co = us["co"]
                self.ex = us["ex"]
                self.ag = us["ag"]
                self.ne = us["ne"]
                self.content_rec_sys_name = us["rec_sys"]
                self.follow_rec_sys_name = us["frec_sys"]
                self.language = us["language"]
                self.owner = us["owner"]
                self.education_level = us["education_level"]
                self.round_actions = us["round_actions"]
                self.joined_on = us["joined_on"]
                self.gender = us["gender"]
                self.toxicity = us["toxicity"]
                self.nationality = us["nationality"]
                self.is_page = us["is_page"]
                self.original_id = us.get("original_id", 0)
                self.toxicity_posts_avg = us.get("toxicity_post_avg", 0)
                self.toxicity_posts_var = us.get("toxicity_post_var", 0)
                self.toxicity_comment = us.get("toxicity_comment", 0)
                self.activity_post = us.get("activity_post", 0)
                self.activity_comment = us.get("activity_comment", 0)
                self.coalition_program = us.get("coalition_program", "")
                self.is_misinfo = us.get("is_misinfo", 0)
                self.susceptibility = us.get("susceptibility", 0)


            config_list = {
                "model": f"{self.type}",
                "base_url": self.llm_base,
                "timeout": 10000,
                "api_type": api_type,
                "api_key": api_key if (api_key is not None and api_key != "") else "NULL",
                "price": [0, 0],
            }

            self.llm_config = {
                "config_list": [config_list],
                "seed": np.random.randint(0, 100000),
                "max_tokens": config['servers']['llm_max_tokens'],
                # max response length, -1 no limits. Imposing limits may lead to truncated responses
                "temperature": config['servers']['llm_temperature'],
            }

            # add and configure the content recsys
            self.content_rec_sys = recsys
            if self.content_rec_sys is not None:
                self.content_rec_sys.add_user_id(self.user_id)

            # add and configure the follow recsys
            self.follow_rec_sys = frecsys
            if self.follow_rec_sys is not None:
                self.follow_rec_sys.add_user_id(self.user_id)

            self.prompts = None

    def __web_init(self, name: str,
        email: str,
        pwd: str = None,
        age: int = None,
        interests: list = None,
        leaning: str = None,
        ag_type="llama3",
        load: bool = False,
        recsys: ContentRecSys = None,
        frecsys: FollowRecSys = None,
        config: dict = None,
        big_five: dict = None,
        language: str = None,
        owner: str = None,
        education_level: str = None,
        joined_on: int = None,
        round_actions: int = 3,
        gender: str = None,
        nationality: str = None,
        toxicity: str = "no",
        api_key: str = "NULL",
        is_page: int = 0,
        *args,
        **kwargs,):

        self.emotions = config["posts"]["emotions"]
        self.actions_likelihood = config["simulation"]["actions_likelihood"]
        self.base_url = config["servers"]["api"]
        self.llm_base = config["servers"]["llm"]
        self.content_rec_sys_name = None
        self.follow_rec_sys_name = None
        self.content_rec_sys = None
        self.follow_rec_sys = None

        self.name = name
        self.email = email
        self.attention_window = int(config["agents"]["attention_window"])

        if "prompts" in kwargs:
            self.prompts = kwargs["prompts"]
            # save on agent custom prompt
            if self.prompts is not None:
                aprompt = Agent_Custom_Prompt(name=self.name, prompt=self.prompts)
                session.add(aprompt)
                session.commit()

        self.llm_v_config = {
            "url": config["servers"]["llm_v"],
            "api_key": config["servers"]["llm_v_api_key"] if (config["servers"]["llm_v_api_key"] is not None and config["servers"]["llm_v_api_key"] != "") else "NULL",
            "temperature": config["servers"]["llm_v_temperature"],
            "max_tokens": int(config["servers"]["llm_v_max_tokens"])
        }
        try:
            self.llm_v_config["model"] = config["servers"]["llm_v_agent"]
        except:
            self.llm_v_config["model"] = 'minicpm-v'

        self.is_page = is_page

        if not load:
            self.language = language
            self.type = ag_type
            self.age = age
            self.interests = interests
            self.leaning = leaning
            self.pwd = pwd
            try:
                self.oe = big_five["oe"]
                self.co = big_five["co"]
                self.ex = big_five["ex"]
                self.ag = big_five["ag"]
                self.ne = big_five["ne"]

            except:
                self.oe = kwargs["oe"]
                self.co = kwargs["co"]
                self.ex = kwargs["ex"]
                self.ag = kwargs["ag"]
                self.ne = kwargs["ne"]

            self.toxicity = toxicity
            self.owner = owner
            self.education_level = education_level
            self.joined_on = joined_on
            sc = SimulationSlot(config)
            sc.get_current_slot()
            self.joined_on = sc.id
            self.round_actions = round_actions
            self.gender = gender
            self.nationality = nationality

            uid = self.__register()
            if uid is None:
                pass
            else:
                self.user_id = uid

        else:
            us = json.loads(self.__get_user())
            self.user_id = us["id"]
            self.type = us["user_type"]
            self.age = us["age"]

            if us["is_page"] == 0:
                try:
                    self.interests = random.randint(config["agents"]["n_interests"]["min"],
                                                    config["agents"]["n_interests"]["max"])
                    self.interests = self._get_interests(-1)[0]
                except:
                    self.interests = interests
                    self.interests = self._get_interests(-1)[0]
            else:
                self.interests = []

            self.leaning = us["leaning"]
            self.pwd = us["password"]
            self.oe = us["oe"]
            self.co = us["co"]
            self.ex = us["ex"]
            self.ag = us["ag"]
            self.ne = us["ne"]
            self.content_rec_sys_name = us["rec_sys"]
            self.follow_rec_sys_name = us["frec_sys"]
            self.language = us["language"]
            self.owner = us["owner"]
            self.education_level = us["education_level"]
            self.round_actions = us["round_actions"]
            self.joined_on = us["joined_on"]
            self.gender = us["gender"]
            self.toxicity = us["toxicity"]
            self.nationality = us["nationality"]
            self.is_page = us["is_page"]

        config_list = {
            "model": f"{self.type}",
            "base_url": self.llm_base,
            "timeout": 10000,
            "api_type": "open_ai",
            "api_key": api_key if (api_key is not None and api_key != "") else "NULL",
            "price": [0, 0],
        }

        self.llm_config = {
            "config_list": [config_list],
            "seed": np.random.randint(0, 100000),
            "max_tokens": int(config['servers']['llm_max_tokens']),
            # max response length, -1 no limits. Imposing limits may lead to truncated responses
            "temperature": float(config['servers']['llm_temperature']),
        }

        self.set_rec_sys(recsys, frecsys)

        # add and configure the content recsys
        if self.content_rec_sys is not None:
            self.content_rec_sys.add_user_id(self.user_id)

        # add and configure the follow recsys
        if self.follow_rec_sys is not None:
            self.follow_rec_sys.add_user_id(self.user_id)

        self.prompts = None

    def _effify(self, non_f_str: str, **kwargs):
        """
        Effify the string.

        :param non_f_str: the string to effify
        :param kwargs: the keyword arguments
        :return: the effified string
        """
        kwargs["self"] = self
        return eval(f'f"""{non_f_str}"""', kwargs)

    def set_prompts(self, prompts):
        """
        Set the LLM prompts.

        :param prompts: the prompts
        """
        self.prompts = prompts

        try:
            # if the agent has custom prompts substitute the default ones
            aprompt = session.query(Agent_Custom_Prompt).filter_by(agent_name=self.name).first()
            if aprompt:
                self.prompts["agent_roleplay"] = f"{aprompt.prompt} - Act as requested by the Handler."
                self.prompts["agent_roleplay_simple"] = f"{aprompt.prompt} - Act as requested by the Handler."
                self.prompts["agent_roleplay_base"] = f"{aprompt.prompt} - Act as requested by the Handler."
                self.prompts["agent_roleplay_comments_share"] = f"{aprompt.prompt} - Act as requested by the Handler."
        except:
            pass

    def set_rec_sys(self, content_recsys, follow_recsys):
        """
        Set the recommendation systems.

        :param content_recsys: the content recommendation system
        :param follow_recsys: the follow recommendation system
        """
        if self.content_rec_sys is None:
            self.content_rec_sys = content_recsys
            self.content_rec_sys.add_user_id(self.user_id)
            self.content_rec_sys_name = content_recsys.name

            api_url = f"{self.base_url}update_user"

            params = {
                "username": self.name,
                "email": self.email,
                "recsys_type": content_recsys.name,
            }
            st = json.dumps(params)
            self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

        if self.follow_rec_sys is None:
            self.follow_rec_sys = follow_recsys
            self.follow_rec_sys.add_user_id(self.user_id)
            self.follow_rec_sys_name = follow_recsys.name

            api_url = f"{self.base_url}update_user"

            params = {
                "username": self.name,
                "email": self.email,
                "frecsys_type": follow_recsys.name,
            }
            st = json.dumps(params)
            self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

        return {"status": 200}
    
    def _extract_components(self, text, c_type="hashtags"):
        """
        Extract the components from the text.

        :param text: the text to extract the components from
        :param c_type: the component type
        :return: the extracted components
        """
        # Define the regex pattern
        if c_type == "hashtags":
            pattern = re.compile(r"#\w+")
        elif c_type == "mentions":
            pattern = re.compile(r"@\w+")
        else:
            return []
        
        # Find all matches and remove duplicates
        components = set(pattern.findall(text))

        # Remove self-mentions if extracting mentions
        if c_type == "mentions":
            components.discard(f"@{self.name}")

        return list(components)
    
    def _remove_components(self, text, components):
        """
        Remove the extracted components from the text.

        :param text: The original text
        :param components: The list of components (hashtags or mentions) to remove
        :return: The cleaned text
        """
        for comp in components:
            text = text.replace(comp, "")
        return text.strip()

    def __get_user(self):
        """
        Get the user from the service.

        :return: the user
        """
        res = json.loads(self._check_credentials())
        if res["status"] == 404:
            raise Exception("User not found")
        api_url = f"{self.base_url}get_user"

        params = {"username": self.name, "email": self.email}
        st = json.dumps(params)

        response = self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

        return response.__dict__["_content"].decode("utf-8")

    def _check_credentials(self):
        """
        Check if the credentials are correct.

        :return: the response from the service
        """
        api_url = f"{self.base_url}user_exists"

        params = {"name": self.name, "email": self.email}

        st = json.dumps(params)
        response = self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

        return response.__dict__["_content"].decode("utf-8")

    def __register(self):
        """
        Register the agent to the service.

        :return: the response from the service
        """

        st = json.dumps(
            {
                "name": self.name,
                "email": self.email,
                "password": self.pwd,
                "leaning": self.leaning,
                "age": self.age,
                "user_type": self.type,
                "oe": self.oe,
                "co": self.co,
                "ex": self.ex,
                "ag": self.ag,
                "ne": self.ne,
                "language": self.language,
                "owner": self.owner,
                "education_level": self.education_level,
                "round_actions": self.round_actions,
                "gender": self.gender,
                "nationality": self.nationality,
                "toxicity": self.toxicity,
                "joined_on": self.joined_on,
                "is_page": self.is_page,
                "original_id": self.original_id,
                "toxicity_post_avg": self.toxicity_posts_avg,
                "toxicity_post_var": self.toxicity_posts_var,
                "toxicity_comment": self.toxicity_comment,
                "activity_post": self.activity_post,
                "activity_comment": self.activity_comment,
                "is_misinfo": self.is_misinfo,
                "susceptibility": self.susceptibility,
            }
        )

        api_url = f"{self.base_url}/register"
        self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

        try:
            res = json.loads(self.__get_user())
            uid = int(res["id"])
        except:
            return None

        api_url = f"{self.base_url}/set_user_interests"
        data = {"user_id": uid, "interests": self.interests, "round": self.joined_on}

        self.session.post(f"{api_url}", headers=API_HEADERS, data=json.dumps(data))

        return uid

    def _get_interests(self, tid, n_max=2):
        if tid == -1:
            # get last round id
            api_url = f"{self.base_url}/current_time"
            response = self.session.get(f"{api_url}", headers=API_HEADERS)
            data = json.loads(response.__dict__["_content"].decode("utf-8"))
            tid = int(data["id"])

        def fetch_interests(time_window):
            api_url = f"{self.base_url}/get_user_interests"
            data = {
                "user_id": self.user_id,
                "round_id": tid,
                "n_interests": self.interests if isinstance(self.interests, int) else len(self.interests),
                "time_window": time_window,
            }
            response = self.session.get(api_url, headers=API_HEADERS, data=json.dumps(data))
            return json.loads(response.content.decode("utf-8"))
        
        try:
            data = fetch_interests(self.attention_window)

            # if no interests in attention window, fetch all
            if not data or len(data) == 0:
                data = fetch_interests(tid)

            # select a random interest without replacement
            if len(data) > n_max:
                selected = np.random.choice(range(len(data)), np.random.randint(1, n_max+1), replace=False)
            else:
                selected = np.random.choice(range(len(data)), len(data), replace=False)

            interests = [data[i]["topic"] for i in selected]
            interests_id = [data[i]["id"] for i in selected]
        except:
            return [], []

        return interests, interests_id
    
    def _get_interests_to_update(self, tid):
        """
        Get the interests the agents interacted with since the last opinion update
        """
        # get round of last opinion
        api_url = f"{self.base_url}/get_last_opinion_round"
        response = self.session.get(f"{api_url}", headers=API_HEADERS, data=json.dumps({"user_id": self.user_id}))
        data = json.loads(response.__dict__["_content"].decode("utf-8"))
        last_opinion_round = int(data["round"])

        # get active topics since last update
        api_url = f"{self.base_url}/get_active_topics"
        data = {
            "user_id": self.user_id,
            "base_round": last_opinion_round,
            "end_round": tid
        }
        response = self.session.get(api_url, headers=API_HEADERS, data=json.dumps(data))
        topics = json.loads(response.content.decode("utf-8"))
        return topics

    def post(self, tid):
        """
        Post a message to the service.

        :param tid: the round id
        """
        topic_descr = "\n".join([f" - {topic.capitalize()}: {TOPIC_DESCRIPTION[topic.lower()]}" for topic in self.interests if topic.lower() in TOPIC_DESCRIPTION])
        current_opinion = opinions_to_str(self.opinions, self.interests)
        coalition_opinion = opinions_to_str(self.coalition_program, self.interests)

        agent_prompt=self._effify(self.prompts["agent_roleplay"], topic_descriptions=topic_descr, coalition_opinion=coalition_opinion, opinion=current_opinion)
        handler_prompt = self.prompts["handler_instructions"]

        interests, interests_id = self._get_interests(tid, n_max=1)
        topic = interests[0] if len(interests) > 0 else ""      

        handler_post_prompt = self.prompts["handler_post_misinfo"] if self.is_misinfo else self.prompts["handler_post"]      
        handler_message = self._effify(handler_post_prompt, topic=topic, toxicity=toxicity_to_label(self.toxicity_posts_avg, self.llm_language))

        api_type = self.llm_config['config_list'][0]['api_type']
        if api_type == "hf":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            post_text = query_huggingface(prompt, llm_config=self.llm_config)
            emotion_eval = []
        elif api_type == "url":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            post_text = query_model(prompt, llm_config=self.llm_config)
            emotion_eval = []
        else:
            _, chat_messages = query_assistant(self.name, self.llm_config, agent_prompt, handler_prompt, handler_message)
            post_text = chat_messages[-2]["content"]
            emotion_eval = chat_messages[-1]["content"].lower()
            emotion_eval = self._clean_emotion(emotion_eval)

        post_text = self._extract_tweet(post_text)

        # avoid posting empty messages
        if len(post_text) < 3:
            return

        # extract hashtags and mentions
        hashtags = self._extract_components(post_text, c_type="hashtags")
        mentions = self._extract_components(post_text, c_type="mentions")

        post_text = self._remove_components(post_text, hashtags)
        post_text = post_text.replace('"', "")
        
        st = json.dumps(
            {
                "user_id": self.user_id,
                "tweet": post_text,
                "emotions": emotion_eval,
                "hashtags": hashtags,
                "mentions": mentions,
                "tid": tid,
                "topics": interests_id,
                "src_language": self.llm_language,
                "tgt_language": self.language
            }
        )

        # send post to server
        api_url = f"{self.base_url}/post"
        self.session.post(f"{api_url}", headers=API_HEADERS, data=st)
        self.memory += f"You posted:\n\"{post_text}\"\n\n"

        # update user interests
        api_url = f"{self.base_url}/set_user_interests"
        data = {"user_id": self.user_id, "interests": interests_id, "round": tid}
        self.session.post(f"{api_url}", headers=API_HEADERS, data=json.dumps(data))

    def _get_thread(self, post_id: int, max_tweets=None):
        """
        Get the thread of a post.

        :param post_id: The post id to get the thread.
        :param max_tweets: The maximum number of tweets to read for context.
        """
        api_url = f"{self.base_url}/post_thread"

        params = {"post_id": post_id}
        st = json.dumps(params)
        response = self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

        res = json.loads(response.__dict__["_content"].decode("utf-8"))
        tweets = res["tweets"]
        latest_post = res["latest_post"]

        if max_tweets is not None and len(res) > max_tweets:
            return tweets[-max_tweets:]

        return tweets, latest_post

    def get_user_from_post(self, post_id: int):
        """
        Get the user from a post.

        :param post_id: The post id to get the user.
        :return: the user
        """
        api_url = f"{self.base_url}/get_user_from_post"

        params = {"post_id": post_id}
        st = json.dumps(params)
        response = self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

        res = json.loads(response.__dict__["_content"].decode("utf-8"))
        return res

    def __get_article(self, post_id: int):
        """
        Get the article.

        :param post_id: The article id to get the article.
        :return: the article
        """
        api_url = f"{self.base_url}/get_article"

        params = {"post_id": int(post_id)}
        st = json.dumps(params)
        response = self.session.post(f"{api_url}", headers=API_HEADERS, data=st)
        if response.status_code == 404:
            return None
        res = json.loads(response.__dict__["_content"].decode("utf-8"))
        return res

    def __get_post(self, post_id: int):
        """
        Get the thread of a post.

        :param post_id: The post id to get the thread.
        :return: the post
        """
        api_url = f"{self.base_url}/get_post"

        params = {"post_id": post_id}
        st = json.dumps(params)
        response = self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

        if response.status_code != 200:
            print(f"\n\nError in getting root for post {post_id}: {response.status_code}, {response.text}\nInterests not updated.\n\n")
            return None

        res = json.loads(response.__dict__["_content"].decode("utf-8"))
        return res

    def comment(self, post_id: int, tid, max_length_threads=None):
        """
        Generate a comment to an existing post

        :param post_id: the post id
        :param tid: the round id
        :param max_length_threads: the maximum length of the thread to read for context
        """
        topic_descr = "\n".join([f" - {topic.capitalize()}: {TOPIC_DESCRIPTION[topic.lower()]}" for topic in self.interests if topic.lower() in TOPIC_DESCRIPTION])
        current_opinion = opinions_to_str(self.opinions, self.interests)
        coalition_opinion = opinions_to_str(self.coalition_program, self.interests)

        agent_prompt=self._effify(self.prompts["agent_roleplay"], opinion=current_opinion, coalition_opinion=coalition_opinion, topic_descriptions=topic_descr)
        handler_prompt = self._effify(self.prompts["handler_instructions"])

        conversation, latest_posts = self._get_thread(post_id, max_tweets=max_length_threads)
        conv = "".join(conversation)

        api_url = f"{self.base_url}/get_post_topics"
        response = self.session.get(f"{api_url}", headers=API_HEADERS,data=json.dumps({"post_id": post_id}))
        post_topics = json.loads(response.__dict__["_content"].decode("utf-8"))

        handler_comment_prompt = self.prompts["handler_comment_misinfo"] if self.is_misinfo else self.prompts["handler_comment"]
        handler_message = self._effify(handler_comment_prompt, topic=post_topics[0]["name"], toxicity=toxicity_to_label(self.toxicity_comment, self.llm_language), conv=conv)
        
        api_type = self.llm_config['config_list'][0]['api_type']
        if api_type == "hf":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            post_text = query_huggingface(prompt, llm_config=self.llm_config)
            emotion_eval = []
        elif api_type == "url":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            post_text = query_model(prompt, llm_config=self.llm_config)
            emotion_eval = []
        else:
            _, chat_messages = query_assistant(self.name, self.llm_config, agent_prompt, handler_prompt, handler_message)
            post_text = chat_messages[-2]["content"]
            emotion_eval = chat_messages[-1]["content"].lower()
            emotion_eval = self._clean_emotion(emotion_eval)
        
        hashtags = self._extract_components(post_text, c_type="hashtags")
        mentions = self._extract_components(post_text, c_type="mentions")

        # avoid posting empty messages
        if len(post_text) < 3 or len(mentions) == 0:
            return
        
        # Cleanup text
        post_text = self._remove_components(post_text, hashtags)
        post_text = self._remove_components(post_text, mentions)
        post_text = self._clean_text(post_text)
        
        # Define tartget post id
        target_post = post_id
        if len(mentions) == 1:
            try:
                target_username = mentions[0].replace("@", "")
                target_post = latest_posts[target_username]
            except:
                pass

        st = json.dumps(
            {
                "user_id": self.user_id,
                "post_id": target_post,
                "text": post_text,
                "emotions": emotion_eval,
                "hashtags": hashtags,
                "mentions": mentions,
                "tid": tid,
                "src_language": self.llm_language,
                "tgt_language": self.language
            }
        )

        api_url = f"{self.base_url}/comment"
        self.session.post(f"{api_url}", headers=API_HEADERS, data=st)
        self.memory += f"You read the thread:\n{conv}\n\n"
        formatted_comment = f"{' '.join(mentions)} {post_text} {' '.join(hashtags)}".strip()
        self.memory += f"You commented:\n\"{formatted_comment}\"\n\n"

        # update topic of interest with the ones from the post
        api_url = f"{self.base_url}/get_thread_root"
        response = self.session.get(f"{api_url}", headers=API_HEADERS, data=json.dumps({"post_id": post_id}))
        if response.status_code != 200:
            print(f"\n\nError in getting root for post {post_id}: {response.status_code}, {response.text}\nInterests not updated.\n\n")
            return 
        root_post_id = response.text
        self._update_user_interests(root_post_id, tid)

    def _update_user_interests(self, post_id, tid):
        """
        Update the user interests based on the post topics.

        :param post_id: id of the post
        :param tid: round id
        """
        api_url = f"{self.base_url}/get_post_topics"
        response = self.session.get(f"{api_url}", headers=API_HEADERS, data=json.dumps({"post_id": post_id}))
        data = json.loads(response.__dict__["_content"].decode("utf-8"))
        topic_ids = [t["id"] for t in data]

        if len(data) > 0:
            api_url = f"{self.base_url}/set_user_interests"
            data = {"user_id": self.user_id, "interests": topic_ids, "round": tid}
            self.session.post(f"{api_url}", headers=API_HEADERS, data=json.dumps(data))

    def share(self, post_id: int, tid):
        """
        Share a post containing a news article.

        :param post_id: the post id
        :param tid: the round id
        :return: the response from the service
        """

        article = self.__get_article(post_id)
        if "status" in article:
            return

        post_text = self.__get_post(post_id)

        # obtain the most recent (and frequent) interests of the agent
        # interests, _ = self.__get_interests(tid)

        # get the post_id topics
        api_url = f"{self.base_url}/get_post_topics"
        response = self.session.get(f"{api_url}", headers=API_HEADERS, data=json.dumps({"post_id": post_id}))
        topics = json.loads(response.__dict__["_content"].decode("utf-8"))
        interests = [t["name"] for t in topics]

        # get the opinion on the topics (if present)
        self.topics_opinions = ""
        if len(interests) > 0:
            # get recent sentiment on the selected interests
            api_url = f"{self.base_url}/get_sentiment"
            data = {"user_id": self.user_id, "interests": interests}
            response = self.session.post(f"{api_url}", headers=API_HEADERS, data=json.dumps(data))
            sentiment = json.loads(response.__dict__["_content"].decode("utf-8"))

            self.topics_opinions = "Your opinion topics of the post you are responding to are: "
            for s in sentiment:
                self.topics_opinions += f"{s['topic']}: {s['sentiment']} "
            if len(sentiment) == 0:
                self.topics_opinions = ""
        else:
            interests, _ = self._get_interests(tid)

        api_type = self.llm_config['config_list'][0]['api_type']

        agent_prompt = self._effify(self.prompts["agent_roleplay_comments_share"], interest=interests)
        handler_prompt = self._effify(self.prompts["handler_instructions"])
        handler_message = self._effify(self.prompts["handler_share"], article=article, post_text=post_text)
        
        if api_type == "hf":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            post_text = query_huggingface(prompt, llm_config=self.llm_config)
            emotion_eval = []
        elif api_type == "url":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            post_text = query_model(prompt, llm_config=self.llm_config)
            emotion_eval = []
        else:
            _, chat_messages = query_assistant(self.name, self.llm_config, agent_prompt, handler_prompt, handler_message)
            post_text = chat_messages[-2]["content"]

            emotion_eval = chat_messages[-1]["content"].lower()
            emotion_eval = self._clean_emotion(emotion_eval)
            
        post_text = (
            post_text.split(":")[-1]
            .split("-")[-1]
            .replace("@ ", "")
            .replace("  ", " ")
            .replace(". ", ".")
            .replace(" ,", ",")
            .replace("[", "")
            .replace("]", "")
            .replace("@,", "")
        )
        post_text = post_text.replace(f"@{self.name}", "")

        hashtags = self._extract_components(post_text, c_type="hashtags")
        mentions = self._extract_components(post_text, c_type="mentions")

        st = json.dumps(
            {
                "user_id": self.user_id,
                "post_id": post_id,
                "text": post_text.replace('"', ""),
                "emotions": emotion_eval,
                "hashtags": hashtags,
                "mentions": mentions,
                "tid": tid,
            }
        )

        api_url = f"{self.base_url}/share"
        self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

    def reaction(self, post_id: int, tid: int, check_follow=True, agent_prompt=None):
        """
        Generate a reaction to a post/comment.

        :param post_id: the post id
        :param tid: the round id
        :param check_follow: whether to evaluate a follow cascade action
        :return: the response from the service
        """
        api_url = f"{self.base_url}/get_post_author"
        response = self.session.get(f"{api_url}", headers=API_HEADERS, data=json.dumps({"post_id": post_id}))
        post_author = json.loads(response.__dict__["_content"].decode("utf-8"))
        if post_author == self.user_id:
            return

        post_text = self.__get_post(post_id)
        if post_text is None:
            return
        
        topic_descr = "\n".join([f" - {topic.capitalize()}: {TOPIC_DESCRIPTION[topic.lower()]}" for topic in self.interests if topic.lower() in TOPIC_DESCRIPTION])
        current_opinion = opinions_to_str(self.opinions, self.interests)
        coalition_opinion = opinions_to_str(self.coalition_program, self.interests)
        
        agent_prompt = self._effify(self.prompts["agent_roleplay"], topic_descriptions=topic_descr, coalition_opinion=coalition_opinion, opinion=current_opinion) if agent_prompt is None else agent_prompt
        handler_prompt = self._effify(self.prompts["handler_instructions_simple"])
        handler_message = self._effify(self.prompts["handler_reactions"], post_text=post_text)

        api_type = self.llm_config['config_list'][0]['api_type']
        if api_type == "hf":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            text = query_huggingface(prompt, llm_config=self.llm_config)
        elif api_type == "url":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            text = query_model(prompt, llm_config=self.llm_config)
        else:
            chat_messages, _ = query_assistant(self.name, self.llm_config, agent_prompt, handler_prompt, handler_message)
            text = chat_messages[-2]["content"].replace("!", "")

        text = text.split()
        if not (("YES" in text) ^ ("NO" in text)):  
            return
        
        reaction = "like" if "YES" in text else "dislike"
        
        st = json.dumps(
                {
                    "user_id": self.user_id,
                    "post_id": post_id,
                    "type": reaction,
                    "tid": tid,
                }
            )

        # add reaction
        api_url = f"{self.base_url}/reaction"
        self.session.post(f"{api_url}", headers=API_HEADERS, data=st)
        self.memory += f"\n\nYou reacted with a '{reaction.upper()}' to the following tweet:\n\"{post_text}\"\n\n"

        # update user interests after reaction
        self._update_user_interests(post_id, tid)
        
        # get follow relationship
        target_user = self.get_user_from_post(post_id)
        data = {"user_id": self.user_id, "target": target_user}

        api_url = f"{self.base_url}/follow_status"
        response = self.session.post(f"{api_url}", headers=API_HEADERS, data=json.dumps(data))
        follow_status = json.loads(response.__dict__["_content"].decode("utf-8"))["status"]

        # evaluate follow/unfollow
        if "YES" in text and follow_status != "follow" and check_follow:
            self.evaluate_follow(post_text, post_id, "follow", tid)
        elif "NO" in text and follow_status == "follow":
            self.evaluate_follow(post_text, post_id, "unfollow", tid)

    def evaluate_follow(self, post_text, post_id, action, tid, agent_prompt=None):
        """
        Evaluate a follow action.

        :param post_text: the post text
        :param post_id: the post id
        :param action: the action, either follow or unfollow
        :param tid: the round id
        :param action_descr: the action description to include in the prompt
        :return: the response from the service
        """
        api_url = f"{self.base_url}/get_post_author"
        response = self.session.get(f"{api_url}", headers=API_HEADERS, data=json.dumps({"post_id": post_id}))
        post_author = json.loads(response.__dict__["_content"].decode("utf-8"))
        if post_author == self.user_id:
            return
        
        locale = LOCALES.get(self.llm_language.lower())
        action_loc = TRANSLATIONS[locale].get(action, action)

        topic_descr = "\n".join([f" - {topic.capitalize()}: {TOPIC_DESCRIPTION[topic.lower()]}" for topic in self.interests if topic.lower() in TOPIC_DESCRIPTION])
        current_opinion = opinions_to_str(self.opinions, self.interests)
        coalition_opinion = opinions_to_str(self.coalition_program, self.interests)

        api_type = self.llm_config['config_list'][0]['api_type']
        agent_prompt = self._effify(self.prompts["agent_roleplay"], topic_descriptions=topic_descr, coalition_opinion=coalition_opinion, opinion=current_opinion) if agent_prompt is None else agent_prompt
        handler_prompt = self._effify(self.prompts["handler_instructions_simple"])
        handler_message = self._effify(self.prompts["handler_follow"], post_text=post_text, action=action_loc)
        
        if api_type == "hf":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            text = query_huggingface(prompt, llm_config=self.llm_config)
        elif api_type == "url":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            text = query_model(prompt, llm_config=self.llm_config)
        else:
            chat_messages, _ = query_assistant(self.name, self.llm_config, agent_prompt, handler_prompt, handler_message)
            text = chat_messages[-2]["content"].replace("!", "")

        if "YES" in text.split():
            self.follow(post_id=post_id, action=action, tid=tid)
            self.memory += f"You decided to {action.upper()} the author of the following tweet:\n\"{post_text}\"\n\n"
            return action
        else:
            return None

    def follow(self, tid: int, target_id: int = None, post_id: int = None, action="follow"):
        """
        Follow a user

        :param tid: the round id
        :param action: the action, either follow or unfollow
        :param post_id: the post id
        :param target: the target user id
        """
        if post_id is not None:
            target_id = self.get_user_from_post(post_id)

        st = json.dumps(
            {
                "user_id": self.user_id,
                "target": int(target_id),
                "action": action,
                "tid": tid,
            }
        )

        api_url = f"{self.base_url}/follow"
        self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

    def followers(self):
        """
        Get the followers of the user.

        :return: the response from the service
        """

        st = json.dumps({"user_id": self.user_id})

        api_url = f"{self.base_url}/followers"
        response = self.session.get(f"{api_url}", headers=API_HEADERS, data=st)

        return response.__dict__["_content"].decode("utf-8")

    def timeline(self):
        """
        Get the timeline of the user.

        :return: the response from the service
        """

        st = json.dumps({"user_id": self.user_id})

        api_url = f"{self.base_url}/timeline"
        response = self.session.get(f"{api_url}", headers=API_HEADERS, data=st)

        return response.__dict__["_content"].decode("utf-8")

    def cast(self, post_id: int, tid: int):
        """
        Cast a voting intention (political simulation)

        :param post_id: the post id
        :param tid: the round id
        :return: the response from the service
        """

        post_text = self.__get_post(post_id)

        topic_descr = "\n".join([f" - {topic.capitalize()}: {TOPIC_DESCRIPTION[topic.lower()]}" for topic in self.interests if topic.lower() in TOPIC_DESCRIPTION])
        current_opinion = opinions_to_str(self.opinions, self.interests)
        coalition_opinion = opinions_to_str(self.coalition_program, self.interests)
        api_type = self.llm_config['config_list'][0]['api_type']

        agent_prompt = self._effify(self.prompts["agent_roleplay"], topic_descriptions=topic_descr, coalition_opinion=coalition_opinion, opinion=current_opinion)
        handler_prompt = self._effify(self.prompts["handler_instructions_simple"])
        handler_message = self._effify(self.prompts["handler_cast"], post_text=post_text)

        if api_type == "hf":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            text = query_huggingface(prompt, llm_config=self.llm_config)
        elif api_type == "url":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            text = query_model(prompt, llm_config=self.llm_config)
        else:
            chat_messages, _ = query_assistant(self.name, self.llm_config, agent_prompt, handler_prompt, handler_message)
            text = chat_messages[-2]["content"].replace("!", "").upper()

        data = {
            "user_id": self.user_id,
            "post_id": post_id,
            "content_type": "Post",
            "tid": tid,
            "content_id": post_id,
        }

        if "RIGHT" in text.split():
            data["vote"] = "R"
            st = json.dumps(data)

        elif "LEFT" in text.split():
            data["vote"] = "D"
            st = json.dumps(data)

        elif "NONE" in text.split():
            data["vote"] = "U"
            st = json.dumps(data)
        else:
            return

        api_url = f"{self.base_url}/cast_preference"
        self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

    def churn_system(self, tid):
        """
        Leave the system.

        :return:
        """
        st = json.dumps({"user_id": self.user_id, "left_on": tid})

        api_url = f"{self.base_url}/churn"
        response = self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

        return response.__dict__["_content"].decode("utf-8")

    def select_action(self, tid, actions, max_length_thread_reading=5):
        """
        Post a message to the service.

        :param actions: The list of actions to select from.
        :param tid: The time id.
        :param max_length_thread_reading: The maximum length of the thread to read.
        """
        np.random.shuffle(actions)
        acts = ",".join(actions)

        agent_prompt = self._effify(self.prompts["agent_roleplay_base"])
        handler_prompt = self._effify(self.prompts["handler_instructions_simple"])
        handler_message = self._effify(self.prompts["handler_action"], actions=acts)        

        api_type = self.llm_config['config_list'][0]['api_type']

        if api_type == "hf":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            text = query_huggingface(prompt, llm_config=self.llm_config)
        elif api_type == "url":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            text = query_model(prompt, llm_config=self.llm_config)
        else:
            chat_messages, _ = query_assistant(self.name, self.llm_config, agent_prompt, handler_prompt, handler_message)
            text = chat_messages[-2]["content"]

        text = text.replace("!", "").upper()

        if "COMMENT" in text.split():
            candidates = json.loads(self.read())
            if len(candidates) > 0:
                selected_post = random.sample(candidates, 1)
                self.comment(
                    int(selected_post[0]),
                    max_length_threads=max_length_thread_reading,
                    tid=tid,
                )
                self.reaction(int(selected_post[0]), check_follow=False, tid=tid)

        elif "POST" in text.split():
            self.post(tid=tid)

        elif "READ" in text.split():
            candidates = json.loads(self.read())
            try:
                selected_post = random.sample(candidates, 1)
                self.reaction(int(selected_post[0]), tid=tid)
            except:
                pass

        elif "SEARCH" in text.split():
            candidates = json.loads(self.search())
            if "status" not in candidates and len(candidates) > 0:
                selected_post = random.sample(candidates, 1)
                self.comment(
                    int(selected_post[0]),
                    max_length_threads=max_length_thread_reading,
                    tid=tid,
                )
                self.reaction(int(selected_post[0]), check_follow=False, tid=tid)

        elif "FOLLOW" in text.split():
            self.search_and_follow(tid)

        elif "SHARE" in text.split():
            candidates = json.loads(self.read(article=True))
            if len(candidates) > 0:
                selected_post = random.sample(candidates, 1)
                self.share(int(selected_post[0]), tid=tid)

        elif "CAST" in text.split():
            candidates = json.loads(self.read())
            try:
                selected_post = random.sample(candidates, 1)
                self.cast(int(selected_post[0]), tid=tid)
            except:
                pass

        elif "IMAGE" in text.split():
            image, article_id = self.select_image(tid=tid)
            if image is not None:
                self.comment_image(image, tid=tid, article_id=article_id)

        return

    def select_action_prob(self, tid, max_length_thread_reading=5, min_read=0):
        """
        Select the action to do based on the activity likelihood.

        :param actions: The list of actions to select from.
        :param tid: The time id.
        :param max_length_thread_reading: The maximum length of the thread to read.
        """
        p_post = self.activity_post
        p_comment = self.activity_comment
        total =  p_comment + p_post
        p_read = max(min_read, 1 - total)
        
        if total > 1 - min_read:
            scale = (1 - min_read) / total
            p_post *= scale
            p_comment *= scale
            p_read = min_read
        
        action = random.choices(["comment", "post", "read"], weights=[p_comment, p_post, p_read])[0]
        self.perform_action(action, tid, max_length_thread_reading)
    
    def perform_action(self, action, tid, max_length_thread_reading=5):
        candidates = json.loads(self.read())

        if len(candidates) == 0 or action=="post":
            self.post(tid=tid)

        elif action == "comment":
            selected_post = random.sample(candidates, 1)
            self.comment(int(selected_post[0]), max_length_threads=max_length_thread_reading, tid=tid)
            self.reaction(int(selected_post[0]), tid=tid)

        elif action=="read":
            selected_post = random.sample(candidates, 1)
            self.reaction(int(selected_post[0]), tid=tid)

    def init_opinions(self):
        api_url = f"{self.base_url}/init_opinions"
        self.session.post(f"{api_url}", headers=API_HEADERS, data=json.dumps({"user_id": self.user_id, "coalition": self.leaning}))
        self._fetch_opinions()
        self.opinions = self.coalition_program

    def update_opinions(self, tid):
        if self.memory == "":
            return
        
        interest_names = self._get_interests_to_update(tid)
        llm_scores = self.evaluate_opinion(interest_names)
        updated_topics = list(llm_scores.keys())

        try:
            # Update with API
            data = {
                "user_id": self.user_id, 
                "interests": updated_topics, 
                "tid": tid, 
                "susceptibility": self.susceptibility,
                "method": self.opinion_model,
                "llm_scores": [llm_scores[i]["score"] for i in updated_topics],
                "descriptions": [llm_scores[i]['description'] for i in updated_topics],
            }
            api_url = f"{self.base_url}/update_opinion"
            self.session.post(f"{api_url}", headers=API_HEADERS, data=json.dumps(data))
            self._fetch_opinions()

            # Update agent opinions and memory
            for i in updated_topics:
                self.opinions[i]['label'] = llm_scores[i]['label']
                self.opinions[i]['description'] = llm_scores[i]['description']

            self.memory = ""
        except:
            print(f"\nERROR: user {self.user_id} updating opinions on interests: {interest_names}.")

    def evaluate_opinion(self, interests):
        current_opinions = opinions_to_str(self.opinions, interests)
        coalition_opinion = opinions_to_str(self.coalition_program, interests)
        bias_instructions = BIAS_INSTRUCTIONS["strong_confirmation"] if self.is_misinfo else BIAS_INSTRUCTIONS["base_confirmation"]
        topic_descr = "\n".join([f" - {topic.capitalize()}: {TOPIC_DESCRIPTION[topic.lower()]}" for topic in interests if topic.lower() in TOPIC_DESCRIPTION])

        agent_prompt = self._effify(self.prompts["agent_roleplay"], topic_descriptions=topic_descr, coalition_opinion=coalition_opinion, opinion=current_opinions)
        handler_prompt = self._effify(self.prompts["handler_instructions_simple"])
        handler_message = self._effify(self.prompts["handler_update_opinions"], topics=interests, bias_instructions=bias_instructions)

        api_type = self.llm_config['config_list'][0]['api_type']
        if api_type == "hf":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            text = query_huggingface(prompt, llm_config=self.llm_config)
        elif api_type == "url":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            text = query_model(prompt, llm_config=self.llm_config)
        else:
            chat_messages, _ = query_assistant(self.name, self.llm_config, agent_prompt, handler_prompt, handler_message)
            text = chat_messages[-2]["content"]

        new_opinions = parse_opinions(text, interests)
        updated_opinions = {}
        for opinion in new_opinions:
            topic = opinion['topic']
            new_score = opinion['score']

            updated_opinions[topic] = {
                "score": new_score,
                'label': SCORE_TO_OPINION_TAG[new_score] if new_score in SCORE_TO_OPINION_TAG else '', 
                'description': opinion['description']
            }

        return updated_opinions
    
    def _fetch_opinions(self):
        st = {
            "user_id": self.user_id,
            "interests": self.interests
        }
        api_url = f"{self.base_url}/get_opinions"
        response = self.session.get(f"{api_url}", headers=API_HEADERS, data=json.dumps(st)) 

        try:
            opinions = json.loads(response.__dict__["_content"].decode("utf-8"))
            self.opinions = {op['topic']: {'label': SCORE_TO_OPINION_TAG[op['score_llm']], 'description': op['description']} for op in opinions}
        except:
            self.opinions = {}
    
    def search_and_follow(self, tid):
        candidates = self.get_follow_suggestions()
        if len(candidates) > 0:
            tot = sum([float(v) for v in candidates.values()])
            probs = [v / tot for v in candidates.values()]
            selected = np.random.choice(
                [int(c) for c in candidates],
                p=probs,
                size=1,
            )[0]
            self.follow(tid=tid, target_id=selected, action="follow")

    def reply(self, tid: int, max_length_thread_reading: int = 5):
        """
        Reply to a mention.

        :param tid:
        :param max_length_thread_reading:
        :return:
        """
        selected_post = json.loads(self.read_mentions())
        if "status" not in selected_post:
            self.comment(
                int(selected_post[0]),
                max_length_threads=max_length_thread_reading,
                tid=tid,
            )
        return

    def read(self, article=False):
        """
        Read n_posts from the service.

        :param article: whether to read an article or not
        :return: the response from the service
        """

        return self.content_rec_sys.read(self.base_url, self.user_id, article)

    def read_mentions(self):
        """
        Read n_posts from the service.

        :return: the response from the service
        """
        return self.content_rec_sys.read_mentions(self.base_url)

    def search(self):
        """
        Read n_posts from the service.

        :return: the response from the service
        """
        return self.content_rec_sys.search(self.base_url)

    def get_follow_suggestions(self):
        """
        Read n_posts from the service.

        :return: the response from the service
        """
        return self.follow_rec_sys.follow_suggestions(self.base_url)

    def select_news(self):
        """
        Select a news article from the service.

        :return: the response from the service
        """

        # Select websites with the same leaning of the agent
        candidate_websites = (
            session.query(Websites).filter(Websites.leaning == self.leaning).all()
        )

        # Select a random website
        if len(candidate_websites) == 0:
            candidate_websites = session.query(Websites).all()

        if len(candidate_websites) == 0:
            return "", ""

        # Select a random website from a list
        website = np.random.choice(candidate_websites)

        # Select a random article
        website_feed = NewsFeed(website.name, website.rss)
        website_feed.read_feed()
        article = website_feed.get_random_news()
        return article, website

    def select_image(self, tid):
        """
        Select an image

        :return: the response from the service
        """
        # randomly select an image from database
        image = session.query(Images).order_by(func.random()).first()

        # @Todo: add the case of no news sharing enabled
        if (
            "news" not in self.actions_likelihood
            or self.actions_likelihood["news"] == 0
        ):
            if image is None:
                # where to get the image from??
                return None, None
            else:
                if image.description is not None:
                    return image, None

                else:
                    # annotate the image with a description
                    an = Annotator(config=self.llm_v_config)
                    description = an.annotate(image.url)

                    if description is not None:
                        image.description = description
                        session.commit()
                    else:
                        # delete image
                        session.delete(image)
                        session.commit()
                        return None, None

                    return image, None

        # the news module is active: images will be selected among RSS shared articles
        else:
            # no image available, select a news article and extract image from it
            if image is None:
                news, website = self.select_news()

                if news == "":
                    return None, None

                # get image given article id and set the remote id
                image = (
                    session.query(Images).order_by(func.random()).first()
                )

                if image is None:
                    return None, None
                else:
                    image.remote_article_id = None
                    session.commit()

                    # annotate the image with a description
                    an = Annotator(self.llm_v_config)
                    description = an.annotate(image.url)

                    if description is not None:
                        image.description = description
                        session.commit()
                    else:
                        # delete image
                        session.delete(image)
                        session.commit()
                        return None, None

                    return image, None

            # images available, check if they have a description
            else:
                if image.description is not None:
                    return image, None

                else:
                    # annotate the image with a description
                    an = Annotator(config=self.llm_v_config)
                    description = an.annotate(image.url)
                    if description is not None:
                        image.description = description
                        session.commit()
                    else:
                        # delete image
                        session.delete(image)
                        session.commit()
                        return None, None

                    return image, None

    def comment_image(self, image: object, tid: int, article_id: int = None):
        """
        Comment on an image

        :param image:
        :param tid:
        :param article_id:
        :return:
        """
        # obtain the most recent (and frequent) interests of the agent
        interests, _ = self._get_interests(tid)

        self.topics_opinions = ""

        api_type = self.llm_config['config_list'][0]['api_type']
        agent_prompt = self._effify(self.prompts["agent_roleplay_comments_share"])
        handler_prompt = self._effify(self.prompts["handler_instructions"])
        handler_message = self._effify(self.prompts["handler_comment_image"], descr=image.description)

        if api_type == "hf":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            post_text = query_huggingface(prompt, llm_config=self.llm_config)
            emotion_eval = []
        elif api_type == "url":
            prompt = f"{agent_prompt}\n\n{handler_message}"
            post_text = query_model(prompt, llm_config=self.llm_config)
            emotion_eval = []
        else:
            _, chat_messages = query_assistant(self.name, self.llm_config, agent_prompt, handler_prompt, handler_message)
            post_text = chat_messages[-2]["content"]

            emotion_eval = chat_messages[-1]["content"].lower()
            emotion_eval = self._clean_emotion(emotion_eval)
            

        # cleaning the post text of some unwanted characters
        # post_text = self.__clean_text(post_text)

        # avoid posting empty messages
        if len(post_text) < 3:
            return

        hashtags = self._extract_components(post_text, c_type="hashtags")

        st = json.dumps(
            {
                "user_id": self.user_id,
                "text": post_text.replace('"', "")
                .replace(f"{self.name}", "")
                .replace(":", "")
                .replace("*", ""),
                "emotions": emotion_eval,
                "hashtags": hashtags,
                "tid": tid,
                "image_url": image.url,
                "image_description": image.description,
                "article_id": article_id,
            }
        )

        api_url = f"{self.base_url}/comment_image"
        self.session.post(f"{api_url}", headers=API_HEADERS, data=st)

    def __str__(self):
        """
        Return a string representation of the Agent object.

        :return: the string representation
        """
        return f"Name: {self.name}, Age: {self.age}, Type: {self.type}"

    def __dict__(self):
        """
        Return a dictionary representation of the Agent object.

        :return: the dictionary representation
        """

        interests = self._get_interests(-1, n_max=len(self.interests))

        return {
            "name": self.name,
            "email": self.email,
            "password": self.pwd,
            "age": self.age,
            "type": self.type,
            "leaning": self.leaning,
            "interests": interests,
            "oe": self.oe,
            "co": self.co,
            "ex": self.ex,
            "ag": self.ag,
            "ne": self.ne,
            "rec_sys": self.content_rec_sys_name,
            "frec_sys": self.follow_rec_sys_name,
            "language": self.language,
            "owner": self.owner,
            "education_level": self.education_level,
            "round_actions": self.round_actions,
            "gender": self.gender,
            "nationality": self.nationality,
            "toxicity": self.toxicity,
            "joined_on": self.joined_on,
            "is_page": self.is_page,
            "original_id": getattr(self, 'original_id', 0),
            "toxicity_posts_avg": getattr(self, 'toxicity_posts_avg', 0),
            "toxicity_posts_var": getattr(self, 'toxicity_posts_var', 0),
            "toxicity_comment": getattr(self, 'toxicity_comment', 0),
            "activity_post": getattr(self, 'activity_post', 0),
            "activity_comment": getattr(self, 'activity_comment', 0),
            "reply_coalition": json.loads(json.dumps(getattr(self, 'reply_coalition', {}), default=str)) if hasattr(self, 'reply_coalition') else None,
            "is_misinfo": getattr(self, 'is_misinfo', 0),
            "susceptibility": getattr(self, 'susceptibility', -1),
        }

    def _clean_emotion(self, text):
        try:
            emotion_eval = [
                e.strip()
                for e in text.replace("'", " ")
                .replace('"', " ")
                .replace("*", "")
                .replace(":", " ")
                .replace("[", " ")
                .replace("]", " ")
                .replace(",", " ")
                .replace("-", " ") 
                .replace("\n", " ")
                .split(" ")
                if e.strip() in self.emotions
            ]
        except:
            emotion_eval = []
        return emotion_eval
    
    def _is_valid_tweet(self, text):
        if "I can't fulfill this request" in text:
            return False
        return True

    def _extract_tweet(self, text):
        if not self._is_valid_tweet(text):
            return ""
        
        cleaned_text = re.sub(r"^[\"']|[\"']$", '', text).strip()
        return cleaned_text

    def _clean_text(self, s):        
        s = s.replace(f"@{self.name}", "")
        s = re.sub(r"^[^A-Za-z]+", "", s).strip()
        s = s.strip('"')
        s = re.sub(r"[\-\[\]\*\{\}]", "", s)
        return s.strip()



class Agents(object):
    def __init__(self):
        """
        Initialize the Agent object.
        """
        self.agents = []

    def add_agent(self, agent: Agent):
        """
        Add a profile to the Agents object.

        :param agent: The Profile object to add.
        """
        self.agents.append(agent)

    def remove_agent(self, agent: Agent):
        """
        Remove a profile from the Agents object.

        :param agent: The Profile object to remove.
        """
        self.agents.remove(agent)

    def remove_agent_by_ids(self, agent_ids: list):
        """
        Remove a profile from the Agents object.

        :param agent: The Profile object to remove.
        """
        agent_ids = {int(aid): None for aid in agent_ids}
        for agent in self.agents:
            if agent.user_id in agent_ids:
                self.agents.remove(agent)

    def get_agents(self):
        return self.agents

    def agents_iter(self):
        """
        Iterate over the agents.
        """
        for agent in self.agents:
            yield agent

    def get_ids(self):
        """
        Get the ids of the agents.

        :return: the ids of the agents
        """
        return [p.user_id for p in self.agents]

    def __str__(self):
        """
        Return a string representation of the Agents object.

        :return: the string representation
        """
        return "".join([p.__str__() for p in self.agents])

    def __dict__(self):
        """
        Return a dictionary representation of the Agents object.

        :return: the dictionary representation
        """
        return {"agents": [p.__dict__() for p in self.agents]}

    def __eq__(self, other):
        """
        Return True if the Agents objects are equal.

        :param other: The other agent object to compare.
        :return: True if the Agents objects are equal.
        """
        return self.__dict__() == other.__dict__()
