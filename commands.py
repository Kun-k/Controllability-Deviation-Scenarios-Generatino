from math import pi

def ubuntu_shell(label, argument_list, arg_list):
    conda_activate = "conda activate Scenario_Generation; "
    python_script = "nohup python "
    cd_to_path = "cd /home1/rk19/Scenario_Generation/Controllability-Deviation-Scenarios-Generatino/; "
    pyfile = "main_h2o.py " if label == "h2o" \
        else "main_sac.py " if label == "sac" \
        else "main_cql.py " if label == "cql" \
        else "main.py"
    commands = []
    data_root = "/home1/rk19/Scenario_Generation/dataset/Re_2_H2O/"
    for i in range(len(arg_list)):
        arg_list[i][6] = data_root + arg_list[i][6][int(arg_list[i][3]) - 1]
        cmd_python = python_script + pyfile
        for j in range(len(arg_list[i])):
            cmd_python = cmd_python + argument_list[j] + "=" + arg_list[i][j]
        cmd_python += " >/dev/null 2>&1 &"
        commands.append(cd_to_path + cmd_python)
        print(cd_to_path + conda_activate + cmd_python)
    return commands

def ubuntu_shell_1(label, argument_list, arg_list):
    conda_activate = "conda activate Scenario_Generation; "
    python_script = "nohup python "
    cd_to_path = "cd /home/rrkk/Scenario_Generation/Controllability-Deviation-Scenarios-Generatino/; "
    pyfile = "main_h2o.py " if label == "h2o" \
        else "main_sac.py " if label == "sac" \
        else "main_cql.py " if label == "cql" \
        else "main.py"
    commands = []
    data_root = "/disk/rrkk/scenario_generation/dataset/Re_2_H2O/"
    for i in range(len(arg_list)):
        arg_list[i][6] = data_root + arg_list[i][6][int(arg_list[i][3]) - 1]
        cmd_python = python_script + pyfile
        for j in range(len(arg_list[i])):
            cmd_python = cmd_python + argument_list[j] + "=" + arg_list[i][j]
        cmd_python += " >/dev/null 2>&1 &"
        commands.append(cd_to_path + cmd_python)
        print(cd_to_path + conda_activate + cmd_python)
    return commands

def windows_cmd(label, argument_list, arg_list):
    pyfile = "main_h2o.py " if label == "h2o" \
        else "main_sac.py " if label == "sac" \
        else "main_cql.py " if label == "cql" \
        else "main.py"
    data_root = "E:/scenario_generation/dataset/Re_2_H2O/"
    for i in range(len(arg_list)):
        arg_list[i][6] = data_root + arg_list[i][6][int(arg_list[i][3]) - 1]
        cmd = "conda activate scenario_generation && " \
              "E: && " \
              "cd E:/scenario_generation/Controllability-Deviation-Scenarios-Generatino/ && " \
              "python " + pyfile
        for j in range(len(arg_list[i])):
            cmd = cmd + argument_list[j] + "=" + arg_list[i][j]
        # cmd += " >/dev/null 2>&1 &"
        print(cmd)


if __name__ == "__main__":
    r_list = [
        "r2-0-2-5",
        "r2-0-2-10",
        "r2-0-3-10",
        "r2-1-2-5",
        "r2-3-2-5",
        "r2-5-2-5",
    ]

    data_map = [
        "r3_dis_10_car_2/",
        "r3_dis_20_car_3/",
        "r3_dis_20_car_4/",
        "r3_dis_25_car_5/",
        "r3_dis_25_car_6/",
        "r3_dis_25_car_7/"
    ]

    argument_list_sac = [" --USED_wandb", " --ego_policy", " --adv_policy", " --num_agents",
                         " --r_ego", " --r_adv", " --realdata_path",
                         " --is_save", " --device", " --seed",
                         " --deviation_theta", " --alpha_l1", " --batch_ratio"]

    arg_list_sac = [
        ["True", "sumo", "RL", "3", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "0"],
        ["True", "sumo", "RL", "3", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "1"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "2", "0"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "100", "0"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "500", "0"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:1", "1000", "10", "0"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:1", "100", "10", "0"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:1", "10", "10", "0"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:1", "1", "10", "0"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "1"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "2"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "5"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "10"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "100"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "500"],
        # ["True", "sumo", "RL", "6", "r1", "r1", data_map, "False", "cuda:0", "42", "10", "1000"],
    ]

    argument_list_cql = [" --USED_wandb", " --ego_policy", " --adv_policy", " --num_agents",
                         " --r_ego", " --r_adv", " --realdata_path",
                         " --is_save", " --device", " --seed", " --alpha_l2"]

    arg_list_cql = [
        ["True", "sumo", "RL", "3", "r1", "r1", data_map, "False", "cuda:0", "42", "0"],
        ["True", "sumo", "RL", "3", "r1", "r1", data_map, "False", "cuda:0", "42", "1"],
        ["True", "sumo", "RL", "3", "r1", "r1", data_map, "False", "cuda:0", "42", "5"],
        ["True", "sumo", "RL", "3", "r1", "r1", data_map, "False", "cuda:0", "42", "10"],
        ["True", "sumo", "RL", "3", "r1", "r1", data_map, "False", "cuda:0", "42", "100"],
        ["True", "sumo", "RL", "3", "r1", "r1", data_map, "False", "cuda:0", "42", "1000"],
    ]
    # ubuntu_shell("sac", argument_list_sac, arg_list_sac)
    # ubuntu_shell("cql", argument_list_cql, arg_list_cql)
    # ubuntu_shell("", argument_list, arg_list)
    windows_cmd("sac", argument_list_sac, arg_list_sac)

