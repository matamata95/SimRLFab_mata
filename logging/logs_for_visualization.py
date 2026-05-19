import os
import pandas as pd
import numpy as np
from py import log


def load_logs(log_path, log_cols):
    log_df = pd.read_csv(log_path)
    log_df = log_df[log_cols]
    return log_df

def get_sim_time(log_df):
    """
    Sim time is contained in the episode log, but it is incorrect as it doesn't
    match the sim time in the agent reward log. The sim time in the agent reward
    log is correct.
    Input: 
        log_df: pandas dataframe containing the agent reward log.
    Output:
        episode_info: dict containing:
            -episode_counter
            -(sim_start, sim_end)
        for each episode.
    """
    log_df = log_df[["episode", "sim_step", "sim_time"]].dropna()
    log_df.drop(index=0, inplace=True)  # dropping the first value since its episode "0", other indexes remain the same

    num_steps = int(max(log_df[["sim_step"]].values)[0])  # returns the maximum sim step value in a list.
    num_episodes = int(max(log_df[["episode"]].values)[0])  # returns the maximum episode value in a list.

    episode_info = {}

    # The end of the first episode is on the index 100 and subsequent episodes are on the += 100 index.
    # The last episode might not contain 100 sim steps, so we check it seperately.

    # Adding the first episode manually, since index 0 is not in the dict.
    sim_start = 0
    sim_end = log_df["sim_time"][num_steps]
    episode_counter = 1
    episode_info[episode_counter] = (sim_start, sim_end)

    for episode_counter in range(2, num_episodes):
        sim_start = log_df["sim_time"][num_steps * (episode_counter - 1)]
        sim_end = log_df["sim_time"][num_steps * episode_counter]
        episode_info[episode_counter] = (sim_start, sim_end)

    # Checking the last episode
    sim_start = log_df["sim_time"][num_steps * (num_episodes - 1)]
    sim_end = log_df["sim_time"].iloc[-1]
    episode_info[num_episodes] = (log_df["sim_time"][num_steps * (num_episodes - 1)], log_df["sim_time"].iloc[-1])
    return episode_info

def get_mach_actions(log_df):
    COLUMNS_OF_INTEREST  = ["idle_starvation", "processing", "breakdown"]
    mach_actions = {}
    print("Processing machine actions from machine_log...")
    for i in range(MACHINE_NUM):
        action_col = f"machine_{i}_action"
        duration_col = f"machine_{i}_duration"

        df = log_df[[action_col, duration_col]].dropna()
        df = df[df[action_col].isin(COLUMNS_OF_INTEREST)]

        df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
        summary = df.groupby(action_col)[duration_col].agg(count="count", total_duration="sum")
        mach_actions[f"machine_{i}"] = summary
            
    print("Finished processing machine actions.")
    return mach_actions

def get_mach_utilization_per_episode(log_df, episode_info):
    """
    Input:
        log_df: pandas dataframe containing the machine log.
        episode_info: dict of (episode_counter, (sim_start, sim_end)) for each episode.
    Output:
        mach_utilization: dict containing:
            -machine_i
                -list of (episode_counter, utilization, broken, idle) for each episode.
    """
    COLUMNS_OF_INTEREST  = ["idle_starvation", "processing", "breakdown"]
    mach_utilization = {}
    print("Calculating machine utilization per episode...")
    for i in range(MACHINE_NUM):
        action_col = f"machine_{i}_action"
        sim_time_col = f"machine_{i}_sim_time"
        duration_col = f"machine_{i}_duration"

        # Temporary dict containing all actions and durations for the current machine.
        df = log_df[[action_col, sim_time_col, duration_col]].dropna()

        df = df[df[action_col].isin(COLUMNS_OF_INTEREST)]
        df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")

        # Machine log does not contain episode information, just sim_time.
        # Need to extract machine utilization for each episode based on sim_time
        # For each episode.
        episode_utilization = []
        for episode_counter, (sim_start, sim_end) in episode_info.items():
            episode_df = df[(df[sim_time_col] >= sim_start) & (df[sim_time_col] <= sim_end)]
            processing_time = episode_df[episode_df[action_col] == "processing"][duration_col].sum()
            broken_time = episode_df[episode_df[action_col] == "breakdown"][duration_col].sum()
            idle_time = episode_df[episode_df[action_col] == "idle_starvation"][duration_col].sum()
            total_time = processing_time + broken_time + idle_time

            utilization = processing_time / total_time if total_time > 0 else 0
            broken = broken_time / total_time if total_time > 0 else 0
            idle = idle_time / total_time if total_time > 0 else 0
            episode_utilization.append((episode_counter, utilization, broken, idle))
        mach_utilization[f"machine_{i}"] = episode_utilization
    print("Finished calculating machine utilization per episode.")
    return mach_utilization

def get_mach_utilization_all(log_df):
    """
    Input:
        log_df: pandas dataframe containing episode log with mach utils.
    Output:
        mach_utilization: dict containing:
            all machine utilization per episode:
                -list of (episode_counter, utilization, broken, idle, finished_orders, order_wait_time) for each episode.
    """
    COLUMNS_OF_INTEREST  = ["episode_counter", "machines_working", "machines_broken", "machines_idle", "finished_orders", "order_waiting_time"]
    mach_utilization = {}
    print("Calculating machine utilization per episode from episode_log...")
    df = log_df[COLUMNS_OF_INTEREST].fillna(0)

    for episode_counter in range(1, df["episode_counter"].shape[0] + 1):
        episode = episode_counter
        mach_util = df[df["episode_counter"] == episode][["machines_working"]]
        mach_broken = df[df["episode_counter"] == episode][["machines_broken"]]
        mach_idle = df[df["episode_counter"] == episode][["machines_idle"]]
        finished_orders = df[df["episode_counter"] == episode][["finished_orders"]]
        order_wait_time = df[df["episode_counter"] == episode][["order_waiting_time"]]
        mach_utilization[episode] = (mach_util.values[0][0], mach_broken.values[0][0], mach_idle.values[0][0], finished_orders.values[0][0], order_wait_time.values[0][0])
    print("Finished calculating machine utilization per episode from episode_log.")
    return mach_utilization

def get_transp_utilization_per_episode(log_df, episode_info):
    """
    Input:
        log_df: pandas dataframe containing the transport log.
        episode_info: dict of (episode_counter, (sim_start, sim_end)) for each episode.
    Output:
        transp_utilization: dict containing:
            -transport_i
                -list of (episode_counter, working, walking, handling, idle) for each episode.
    """
    COLUMNS_OF_INTEREST = ["idle", "move_to_empty", "pick_up", "put_down", "transport", "waiting_action"]
    transp_utilization = {}
    print("Calculating transport utilization per episode...")
    for i in range(TRANSPORT_NUM):
        action_col = f"transp_{i}_action"
        sim_time_col = f"transp_{i}_sim_time"
        from_col = f"transp_{i}_from_at"
        to_col = f"transp_{i}_to_at"
        duration_col = f"transp_{i}_duration"
        
        df = log_df[[action_col, sim_time_col, from_col, to_col, duration_col]].dropna()
        df = df[df[action_col].isin(COLUMNS_OF_INTEREST)]
        df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
        
        # Transport log does not contain episode information, just sim_time.
        # Need to extract transport utilization for each episode based on sim_time
        # For each episode.

        episode_utilization = {}
        for episode_counter, (sim_start, sim_end) in episode_info.items():
            episode_df = df[(df[sim_time_col] >= sim_start) & (df[sim_time_col] <= sim_end)]
            handling_time = episode_df[episode_df[action_col].isin(["pick_up", "put_down"])][duration_col].sum()
            walking_time = episode_df[episode_df[action_col].isin(["move_to_empty", "transport"])][duration_col].sum()
            idle_time = episode_df[episode_df[action_col].isin(["idle", "waiting_action"])][duration_col].sum()
            working_time = handling_time + walking_time
            total_time = working_time + idle_time

            working = working_time / total_time if total_time > 0 else 0
            walking = walking_time / total_time if total_time > 0 else 0
            handling = handling_time / total_time if total_time > 0 else 0
            idle = idle_time / total_time if total_time > 0 else 0
            episode_utilization[episode_counter] = (working, walking, handling, idle)
        transp_utilization[f"transport_{i}"] = episode_utilization
    print("Finished calculating transport utilization per episode.")
    return transp_utilization

def get_transp_actions(log_df):
    """
    Calculates the total duration and count of transport actions.
    Input:
        log_df: pandas dataframe containing the transport log.
    Output:
        dictionary containing:
            -transport_i
                -action
                    -count
                    -total_duration
    """
    COLUMNS_OF_INTEREST = ["idle", "move_to_empty", "pick_up", "put_down", "transport", "waiting_action"]
    transp_actions = {}
    print("Processing transport actions from transport_log...")
    for i in range(TRANSPORT_NUM):
        action_col = f"transp_{i}_action"
        from_col = f"transp_{i}_from_at"
        to_col = f"transp_{i}_to_at"
        duration_col = f"transp_{i}_duration"

        df = log_df[[action_col, from_col, to_col, duration_col]].dropna()
        df = df[df[action_col].isin(COLUMNS_OF_INTEREST)]

        df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
        summary = df.groupby(action_col)[duration_col].agg(count="count", total_duration="sum")
        transp_actions[f"transport_{i}"] = summary
            
    print("Finished processing transport actions.")
    return transp_actions

def get_agent_reward(log_df):
    """
    """
    COLUMNS_OF_INTEREST = ["episode", "sim_step", "action", "reward"]
    print("Processing agent rewards from agent_reward_log...")


    log_df.drop(index=0, inplace=True)  # dropping the first value since its episode "0", other indexes remain the same

    num_steps = int(max(log_df[["sim_step"]].values)[0])
    num_episodes = int(max(log_df[["episode"]].values)[0])

    agent_reward = {}
    agent_actions = {i: [] for i in range(1, num_episodes + 1)}
    
    df = log_df[COLUMNS_OF_INTEREST].dropna()
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")

    # The last episode might not contain N sim steps, so we check it seperately.
    # As well as the number of actions. We assume that the highest int of a chosen action
    # through all episodes is the number of actions that the agent can choose from 0 indexing.
    NUM_ACTIONS = 0
    for episode_counter in range(num_episodes - 1):
        for step in range(1, num_steps + 1):
            action = df["action"][episode_counter * num_steps + step]
            action = int(action.strip("[]"))

            reward = df["reward"][episode_counter * num_steps + step]
            agent_reward[(episode_counter + 1, step)] = (action, reward)
            agent_actions[(episode_counter + 1)].append(action)

            if action > NUM_ACTIONS:
                NUM_ACTIONS = action

    # Checking the last episode
    last_episode_steps = int(df["sim_step"].iloc[-1])
    for step in range(1, last_episode_steps + 1):
        action = df["action"][(num_episodes) * last_episode_steps + step]
        action = int(action.strip("[]"))

        reward = df["reward"][(num_episodes) * last_episode_steps + step]
        agent_reward[(num_episodes, step)] = (action, reward)

        agent_actions[(num_episodes)].append(action)

        if action > NUM_ACTIONS:
            NUM_ACTIONS = action

    agent_actions_count = aggregate_actions(agent_actions, num_actions=NUM_ACTIONS + 1)        
    agent_actions_count_normalized = normalize_counts(agent_actions_count)

    print("Finished processing agent rewards.")
    return agent_reward, agent_actions, agent_actions_count_normalized

def transp_dict_to_csv(data_dict, save=False, output_path=None):
    all_rows = []
    for transp, episodes in data_dict.items():
        for ep, vals in sorted(episodes.items()):
            # working, walking, waiting, idle = vals
            working, walking, handling, idle = vals
            all_rows.append({
                "transport": transp,
                "episode_counter": int(ep),
                "working": working,
                "walking": walking,
                # "waiting": waiting,
                "handling": handling, # ++
                "idle": idle
            })
    df = pd.DataFrame(all_rows)
    if save and output_path:
        df.to_csv(output_path, index=False)
    return df

def transp_actions_dict_to_csv(data_dict, save=False, output_path=None):
    all_rows = []
    for transp, summary in data_dict.items():
        row = {"transport": transp}
        for action, vals in summary.iterrows():
            row[f"{action}_count"] = int(vals["count"])
            row[f"{action}_total_duration"] = vals["total_duration"]
        all_rows.append(row)
    df = pd.DataFrame(all_rows)
    if save and output_path:
        df.to_csv(output_path, index=False)
    return df

def mach_act_to_csv(data_dict, save=False, output_path=None):
    all_rows = []
    for machine, summary in data_dict.items():
        row = {"machine": machine}
        for action, vals in summary.iterrows():
            row[f"{action}_count"] = int(vals["count"])
            row[f"{action}_total_duration"] = vals["total_duration"]
        all_rows.append(row)
    df = pd.DataFrame(all_rows)
    if save and output_path:
        df.to_csv(output_path, index=False)
    return df

def mach_util_to_csv(data_dict, save=False, output_path=None):
    all_rows = []
    for machine, utilization in data_dict.items():
        for episode_counter, util, broken, idle in utilization:
            all_rows.append({
                "machine": machine,
                "episode_counter": int(episode_counter),
                "utilization": util,
                "broken": broken,
                "idle": idle
            })
    df = pd.DataFrame(all_rows)
    if save and output_path:
        df.to_csv(output_path, index=False)
    return df

def mach_util_all_to_csv(data_dict, save=False, output_path=None):
    """
    Input df is from episode_log.csv.
    """
    all_rows = []
    for episode_counter, utilization in data_dict.items():
        util, broken, idle, finished_orders, order_wait_time = utilization
        all_rows.append({
            "episode_counter": int(episode_counter),
            "utilization": util,
            "broken": broken,
            "idle": idle,
            "finished_orders": finished_orders,
            "order_wait_time": order_wait_time,
        })
    df = pd.DataFrame(all_rows)
    if save and output_path:
        df.to_csv(output_path, index=False)
    return df

def agent_act_to_csv(data_dict, save=False, output_path=None):
    all_rows = []
    for episode, actions in sorted(data_dict.items()):
        for step, action in enumerate(actions, start=1):
            all_rows.append({
                "episode": int(episode),
                "step": step,
                "action": action
            })
    df = pd.DataFrame(all_rows)
    if save and output_path:
        df.to_csv(output_path, index=False)
    return df

def agent_act_count_to_csv(data_dict, save=False, output_path=None):
    row = {f"action_{k}_freq": v for k, v in data_dict.items()}
    df = pd.DataFrame([row])
    if save and output_path:
        df.to_csv(output_path, index=False)
    return df

def agent_reward_to_csv(data_dict, save=False, output_path=None):
    all_rows = []
    for (episode, step), (action, reward) in sorted(data_dict.items()):
        all_rows.append({
            "episode": int(episode),
            "step": int(step),
            "action": action,
            "reward": reward
        })
    df = pd.DataFrame(all_rows)
    if save and output_path:
        df.to_csv(output_path, index=False)
    return df

# UTILITY FUNCTIONS
def aggregate_actions(agent_actions, num_actions=None):
    lists = list(agent_actions.values())
    if not lists:
        return {}
    all_actions = np.concatenate([np.asarray(l, dtype=int) for l in lists])
    if num_actions is None:
        num_actions = int(all_actions.max()) + 1 if all_actions.size else 0
    counts = np.bincount(all_actions, minlength=num_actions)
    return {str(i): int(counts[i]) for i in range(len(counts))}

def normalize_counts(counts_dict):
    keys = sorted(counts_dict.keys(), key=lambda k: int(k))
    vals = np.array([counts_dict[k] for k in keys], dtype=float)
    s = vals.sum()
    if s == 0:
        return {k: 0.0 for k in keys}
    freqs = vals / s
    return {k: round(float(freqs[i]), 4) for i, k in enumerate(keys)}



if __name__ == "__main__":

    LOG_NAME = "47 states viper"
    LOG_PATH = os.path.join("./log/", LOG_NAME)

    OUTPUT_DIR = os.path.join(LOG_PATH, "log_visualization/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TRANSPORT_NUM = 1
    MACHINE_NUM = 8

    agent_cols = [
        "episode", "sim_step", "sim_time", "action", "reward"
    ]
    machine_cols = [x for i in range(MACHINE_NUM) for x in (f"machine_{i}_action", f"machine_{i}_sim_time", f"machine_{i}_duration")]
    transport_cols = [x for i in range(TRANSPORT_NUM) for x in (f"transp_{i}_action", f"transp_{i}_sim_time", f"transp_{i}_from_at", f"transp_{i}_to_at", f"transp_{i}_duration")]
    episode_cols = [
        "episode_counter", "sim_time", "dt", "total_reward", "machines_working",
        "machines_broken", "machines_idle", "processed_orders", "transp_working",
        "transp_walking", "transp_handling", "transp_idle", "finished_orders",
        "order_waiting_time", "alpha", "inventory"
    ]

    LOGS = (
        ("agent_reward_log", agent_cols),
        ("machine_log", machine_cols),
        ("transport_log", transport_cols),
        ("episode_log", episode_cols)
    )

    log_dfs = {}
    for log, cols in LOGS:
        log_path = os.path.join(LOG_PATH, log + ".csv")
        log_dfs[log] = load_logs(log_path, cols)

    episode_info = get_sim_time(log_dfs["agent_reward_log"])
    
    mach_actions = get_mach_actions(log_dfs["machine_log"])
    mach_actions_df = mach_act_to_csv(mach_actions, save=True, output_path=os.path.join(OUTPUT_DIR, "machine_actions.csv"))   
    mach_util = get_mach_utilization_per_episode(log_dfs["machine_log"], episode_info)
    mach_util_df = mach_util_to_csv(mach_util, save=True, output_path=os.path.join(OUTPUT_DIR, "machine_utilization_per_episode.csv"))
    mach_util_all = get_mach_utilization_all(log_dfs["episode_log"])
    mach_util_all_df = mach_util_all_to_csv(mach_util_all, save=True, output_path=os.path.join(OUTPUT_DIR, "machine_utilization_all.csv"))

    transp_actions = get_transp_actions(log_dfs["transport_log"])
    transp_actions_df = transp_actions_dict_to_csv(transp_actions, save=True, output_path=os.path.join(OUTPUT_DIR, "transport_actions.csv"))
    
    transp_util = get_transp_utilization_per_episode(log_dfs["transport_log"], episode_info)
    transp_util_df = transp_dict_to_csv(transp_util, save=True, output_path=os.path.join(OUTPUT_DIR, "transport_utilization_per_episode.csv"))

    agent_reward, agent_actions, agent_actions_count = get_agent_reward(log_dfs["agent_reward_log"])
    agent_actions_df = agent_act_to_csv(agent_actions, save=True, output_path=os.path.join(OUTPUT_DIR, "agent_actions.csv"))
    agent_actions_count_df = agent_act_count_to_csv(agent_actions_count, save=True, output_path=os.path.join(OUTPUT_DIR, "agent_actions_count.csv"))    
    agent_reward_df = agent_reward_to_csv(agent_reward, save=True, output_path=os.path.join(OUTPUT_DIR, "agent_reward.csv"))


    