from collections import OrderedDict
import random
import shutil
import tqdm
import sys
import os
import networkx as nx
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from YClient.coalitions.coalition_utils import get_topic_opinions_and_descriptions
from y_client import Agent, Agents, SimulationSlot
from y_client.recsys import *
from YClient.y_client.generators import generate_user
from YClient.y_client.constants import API_HEADERS
from y_client.news_feeds import Feeds, session, Websites, Articles, Images


class YClientBase(object):
    def __init__(
        self,
        config_filename,
        prompts_filename=None,
        agents_filename=None,
        graph_file=None,
        agents_output="agents.json",
        owner="admin",
    ):
        """
        Initialize the YClient object

        :param config_filename: the configuration file for the simulation in JSON format
        :param prompts_filename: the LLM prompts file for the simulation in JSON format
        :param agents_filename: the file containing the agents in JSON format
        :param graph_file: the file containing the graph of the agents in CSV format, where the number of nodes is equal to the number of agents
        :param agents_output: the file to save the generated agents in JSON format
        :param owner: the owner of the simulation
        """
        if prompts_filename is None:
            raise Exception("Prompts file not found")

        self.prompts = json.load(open(prompts_filename, "r", encoding="utf-8"))
        self.config = json.load(open(config_filename, "r", encoding="utf-8"))
        self.agents_owner = owner
        self.agents_filename = agents_filename
        self.experiment_name = self.config["simulation"]["name"]
        self.experiment_path = f"experiments/{self.experiment_name}"
        self.agents_output = f"{self.experiment_path}/{agents_output}"

        self.days = self.config["simulation"]["days"]
        self.slots = self.config["simulation"]["slots"]
        self.n_agents = self.config["simulation"]["starting_agents"]

        percentage_misinfo_agent = self.config["simulation"]["percentage_misinformation_agents"]
        self.n_misinfo_agents = int(self.n_agents * percentage_misinfo_agent)

        self.percentage_new_agents_iteration = self.config["simulation"][
            "percentage_new_agents_iteration"
        ]
        self.hourly_activity = self.config["simulation"]["hourly_activity"]
        self.percentage_removed_agents_iteration = float(
            self.config["simulation"]["percentage_removed_agents_iteration"]
        )
        self.actions_likelihood = {
            a.upper(): float(v)
            for a, v in self.config["simulation"]["actions_likelihood"].items()
        }
        tot = sum(self.actions_likelihood.values())
        self.actions_likelihood = {
            k: v / tot for k, v in self.actions_likelihood.items()
        }

        # users' parameters
        self.fratio = self.config["agents"]["reading_from_follower_ratio"]
        self.max_length_thread_reading = self.config["agents"]["max_length_thread_reading"]

        # posts' parameters
        self.visibility_rd = self.config["posts"]["visibility_rounds"]

        # initialize simulation clock
        self.sim_clock = SimulationSlot(self.config)

        self.agents = Agents()
        self.feed = Feeds()
        self.content_recsys = None
        self.follow_recsys = None

        if graph_file is not None:
            self.g = nx.read_edgelist(graph_file, delimiter=",", nodetype=int)
            # relabel nodes to start from 0 just in case
            self.g = nx.convert_node_labels_to_integers(self.g, first_label=0)
        else:
            self.g = None

        self.pages = []
        self.misinfo_agents = []

    @staticmethod
    def reset_news_db():
        """
        Reset the news database
        """
        session.query(Articles).delete()
        session.query(Websites).delete()
        session.query(Images).delete()
        session.commit()

    def reset_experiment(self):
        """
        Reset the experiment
        Delete all agents and reset the server database
        """
        api_url = f"{self.config['servers']['api']}reset"
        post(f"{api_url}", headers=API_HEADERS)

    def load_rrs_endpoints(self, filename):
        """
        Load rss feeds from a file

        :param filename: the file containing the rss feeds
        """

        data = json.load(open(filename))
        for f in tqdm.tqdm(data):
            self.feed.add_feed(
                name=f["name"],
                url_feed=f["feed_url"],
                category=f["category"],
                leaning=f["leaning"],
            )

    def set_interests(self):
        """
        Set the interests of the agents
        """
        api_url = f"{self.config['servers']['api']}set_interests"
        data = self.config["agents"]["interests"]
        post(f"{api_url}", headers=API_HEADERS, data=json.dumps(data))

    def set_recsys(self, c_recsys, f_recsys):
        """
        Set the recommendation systems

        :param c_recsys: the content recommendation system
        :param f_recsys: the follower recommendation system
        """
        self.content_recsys = c_recsys
        self.follow_recsys = f_recsys

    def add_agent(self, agent=None, is_misinfo=False):
        """
        Add an agent to the simulation

        :param agent: the agent to add
        """
        if agent is None:
            try:
                agent = generate_user(self.config, owner=self.agents_owner, is_misinformation=is_misinfo)

                if agent is None:
                    return
                agent.set_prompts(self.prompts)
                agent.set_rec_sys(self.content_recsys, self.follow_recsys)
            except Exception:
                pass
        if agent is not None:
            self.agents.add_agent(agent)
            if is_misinfo:
                self.misinfo_agents.append(agent)

    def create_initial_population(self):
        """
        Create the initial population of agents
        """
        # setting global interests
        self.set_interests()

        coalition_file = self.config.get("agents_init", {}).get("coalition_data_file", None)
        user_file = self.config.get("agents_init", {}).get("user_data_file", None)
        min_posts = self.config.get("agents_init", {}).get("min_posts_written", 0)
        min_comments = self.config.get("agents_init", {}).get("min_comments_written", 0)

        if coalition_file is not None:
            print(f"Initializing coalitions from file: {coalition_file}")
            self.init_coalitions_from_data(coalition_file)

        if user_file is not None:
            print(f"Initializing users from file: {user_file}")
            self.init_users_from_data(user_file, min_posts, min_comments)
        else:
            print("Creating initial population of agents")
            for _ in range(self.n_agents - self.n_misinfo_agents):
                self.add_agent()
            
            for _ in range(self.n_misinfo_agents):
                self.add_agent(is_misinfo=True)
    
    def init_coalitions_from_data(self, coalition_file):        
        self.coalition_file = coalition_file
        with open(coalition_file, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        for coalition, data in all_data.items():
            topic_opinions, topic_descriptions = get_topic_opinions_and_descriptions(data["topic_opinion"])

            payload = {
                "coalition": coalition,
                "topic_opinions": topic_opinions,
                "topic_descriptions": topic_descriptions
            }
            api_url = f"{self.config['servers']['api']}set_coalition_opinion"
            post(api_url, headers=API_HEADERS, data=json.dumps(payload))

    def init_users_from_data(self, user_file, min_posts=0, min_comments=0):
        df = pd.read_csv(user_file, encoding='utf-8')
        n_users = self.n_agents-self.n_misinfo_agents
        for _ in tqdm.tqdm(range(n_users), desc="User init", leave=False):
            self.add_agent_from_data(user_file, min_posts, min_comments)

        possible_coalitions = df['coalition'].dropna().unique().tolist()
        random.shuffle(possible_coalitions)
        for i in tqdm.tqdm(range(self.n_misinfo_agents), desc="Misinfo user init", leave=False):
            coalition = possible_coalitions[i % len(possible_coalitions)]
            self.add_misinfo_agent(coalition=coalition)

    def add_agent_from_data(self, user_file, min_posts=0, min_comments=0):
        df = pd.read_csv(user_file, encoding='utf-8')
        
        if min_posts > 0 and min_comments > 0:
            print(f"Selecting users with at least {min_posts} posts and {min_comments} comments")
        user_ids = df[(df["n_posts"] >= min_posts) & (df["n_comments"] >= min_comments)]["userid"].tolist()
        available_user_ids = [u for u in user_ids if u not in self.agents.get_ids()]

        userid = random.choice(available_user_ids)
        user_data = df[df["userid"] == userid]

        try:
            # Generate agent and init from data
            agent = generate_user(
                self.config, 
                owner=self.agents_owner,
                original_id=userid,
                political_leaning=user_data['coalition'].iloc[0],
                toxicity_post_avg=round(user_data['toxicity_post_avg'].iloc[0], 3),
                toxicity_post_var=round(user_data['toxicity_post_var'].iloc[0], 3),
                toxicity_comment=round(user_data['toxicity_comment_avg'].iloc[0], 3),
                n_posts=round(user_data['n_posts'].iloc[0], 3),
                n_comments=round(user_data['n_comments'].iloc[0], 3),
                coalition_file=self.coalition_file,
                )
            if agent is None:
                return

            agent.set_prompts(self.prompts)
            agent.set_rec_sys(self.content_recsys, self.follow_recsys)
            self.add_agent(agent)

        except Exception as e:
            print(f"Error initializing user {userid}: {e}")
            pass

    def add_misinfo_agent(self, coalition):
        agent = generate_user(
                self.config, 
                owner=self.agents_owner,
                is_misinformation=True, 
                political_leaning=coalition,
                coalition_file=self.coalition_file,
            )
        if agent is None:
            return
        agent.set_prompts(self.prompts)
        agent.set_rec_sys(self.content_recsys, self.follow_recsys)
        self.add_agent(agent, is_misinfo=True)

    def load_existing_agents(self, agents_file):
        """
        Load existing agents from a file
        :param agents_file: the JSON file containing the agents
        """
        print(f"Loading agents from file: {agents_file}")
        agents = json.load(open(agents_file, "r"))

        for a in agents["agents"]:
            try:
                ag = Agent(
                    load=True,
                    name=a["name"], email=a["email"], 
                    config=self.config, 
                    api_type=self.config['servers']['llm_api_type']
                )
                ag._fetch_opinions()
                ag.set_prompts(self.prompts)
                ag.set_rec_sys(self.content_recsys, self.follow_recsys)
                self.agents.add_agent(ag)
            except Exception as e:
                print(f"Error loading agent: {a['name']}: {e}\n")
        
        if self.g is not None:
            print(f"Loading agent connections")
            self.load_agent_connections()

    
    def load_agent_connections(self):
        tid, _, _ = self.sim_clock.get_current_slot()
        id_to_agent = {i: agent for i, agent in enumerate(self.agents.agents)}

        for u, v in self.g.edges():
            try:
                fr_a = id_to_agent[u]
                to_a = id_to_agent[v]
                fr_a.follow(tid=tid, target=to_a.user_id)
            except Exception:
                pass

    def save_agents(self):
        """
        Save the agents to a file
        """
        res = self.agents.__dict__()
        json.dump(res, open(self.agents_output, "w"), indent=4)

    def save_experiment(self, tag="save"):
        """
        Save the experiment
        """

        output_folder = f"{self.experiment_path}/{self.experiment_name}_{tag}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # save experiment - client
        for file in os.listdir(self.experiment_path):
            source_item = f"{self.experiment_path}/{file}"
            if os.path.isfile(source_item):
                shutil.copyfile(source_item, f"{output_folder}/{file}")
    
        # save experiment - server
        api_url = f"{self.config['servers']['api']}save_experiment"
        post(f"{api_url}", headers=API_HEADERS, data=json.dumps({"tag": tag}))

    def churn(self, tid):
        """
        Evaluate churn

        :param tid:
        :return:
        """
        if self.percentage_removed_agents_iteration <= 0:
            return
        
        n_users = max(1, int(len(self.agents.agents) * self.percentage_removed_agents_iteration))
        st = json.dumps({"n_users": n_users, "left_on": tid})
        api_url = f"{self.config['servers']['api']}/churn"
        response = post(f"{api_url}", headers=API_HEADERS, data=st)
        data = json.loads(response.__dict__["_content"].decode("utf-8"))["removed"]
        
        self.agents.remove_agent_by_ids(data)

    def print_opinions(self):
        """
        Print the overall sentiment of the agents
        """
        for agent in self.agents.agents:
            print(f"\n--------{agent.name}--------")
            print(agent.get_opinions())
        print()

    def run_simulation(self):
        """
        Run the simulation
        """
        user_file = self.config.get("agents_init", {}).get("user_data_file", None)
        
        _, day_init, slot_init = self.sim_clock.get_current_slot()
        print(f"Running simulation from day {day_init}, slot {slot_init}\n")
        
        for day in tqdm.tqdm(range(day_init, self.days), desc="Days", total=self.days, initial=day_init):
            daily_actives = OrderedDict()
            tid, _, _ = self.sim_clock.get_current_slot()

            start_slot = slot_init if day == day_init else 0  
            for _ in tqdm.tqdm(range(start_slot, self.slots), desc="Slots", leave=False, total=self.slots, initial=start_slot):
                tid, _, h = self.sim_clock.get_current_slot()

                # get expected active users for this time slot (at least 1)
                expected_active_users = max(int(len(self.agents.agents) * self.hourly_activity[str(h)]), 1)

                sagents = random.sample(self.agents.agents, expected_active_users)

                # available actions
                acts = [a for a, v in self.actions_likelihood.items() if v > 0]

                # shuffle agents
                random.shuffle(sagents)
                for g in tqdm.tqdm(sagents, desc=f"Agents", leave=False):
                    daily_actives[g] = None

                    for _ in range(g.round_actions):                                
                        # select action to be performed  
                        if user_file is not None:
                            g.select_action_prob(tid=tid, max_length_thread_reading=self.max_length_thread_reading)
                        else:
                            candidates = random.choices(acts, k=2, weights=[self.actions_likelihood[a] for a in acts])
                            candidates.append("NONE")
                            g.select_action(tid=tid, actions=candidates, max_length_thread_reading=self.max_length_thread_reading)

                        # reply to received mentions
                        if g not in self.pages:
                            g.reply(tid=tid)

                # increment slot
                self.sim_clock.increment_slot()

            # evaluate following (once per day, only for a random sample of daily active agents)
            da = [
                agent for agent in daily_actives
                if agent not in self.pages
                and random.random() < float(self.config["agents"]["probability_of_daily_follow"])
            ]

            if da and len(da) > 0:
                for agent in tqdm.tqdm(da, desc="New friendships", leave=False):
                    if random.random() < 0.5:
                        agent.search_and_follow(tid=tid)

            
            # update opinions of all daily active agents
            for agent in tqdm.tqdm(daily_actives, desc="Updating opinions", leave=False):
                agent.update_opinions(tid=tid)

            total_users = len(self.agents.agents)

            # daily churn
            self.churn(tid)

            # daily new agents
            if self.percentage_new_agents_iteration > 0:
                for _ in range(max(1, int( len(daily_actives) * self.percentage_new_agents_iteration))):
                    if user_file is not None:
                        if random.random() < self.config["simulation"]["percentage_misinformation_agents"]:
                            df = pd.read_csv(user_file, encoding='utf-8')
                            coalition = random.choice(df['coalition'].dropna().unique().tolist())
                            self.add_misinfo_agent(coalition)
                        else: 
                            min_posts = self.config.get("agents_init", {}).get("min_posts_written", 0)
                            min_comments = self.config.get("agents_init", {}).get("min_comments_written", 0)
                            self.add_agent_from_data(user_file, min_posts=min_posts, min_comments=min_comments)
                    else:
                        self.add_agent()

            # saving "living" agents at the end of the day
            if (
                self.percentage_removed_agents_iteration != 0
                or self.percentage_removed_agents_iteration != 0
            ):
                self.save_agents()

            print(f"\nTotal Users: {total_users}")
            print(f"Active users: {len(daily_actives)}")
            if len(self.agents.agents) != total_users:
                print(f"Users at the end of the day: {len(self.agents.agents)}")
            print("\n" + "─" * 60 + "\n")
