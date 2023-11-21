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
    data_root = "../byH2O/dataset/"
    for i in range(len(arg_list)):
        arg_list[i][6] = data_root + arg_list[i][6]
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
        arg_list[i][6] = data_root + arg_list[i][6]
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
        arg_list[i][6] = data_root + arg_list[i][6]
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

    argument_list_sac = [" --USED_wandb", " --ego_policy", " --adv_policy", " --num_agents",
                         " --r_ego", " --r_adv", " --realdata_path",
                         " --is_save", " --device", " --seed", " --deviation_theta",
                         " --alpha_l1"]

    arg_list_sac = [
        ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "10", "0"],
        ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "10", "1000"],
        ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "10", "5000"],
        # ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "10", "500"],
        # ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "60", "10"],
        # ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "60", "100"],

        # ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "0"],
        # ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "10"],
        # ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "30"],
        # ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "50"],
        # ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "70"],
        # ["True", "sumo", "RL", "3", "r1", "r1", "r3_dis_20_car_4/", "False", "cuda:0", "42", "90"],
    ]

    # ubuntu_shell("sac", argument_list_sac, arg_list_sac)
    # ubuntu_shell("", argument_list, arg_list)
    windows_cmd("sac", argument_list_sac, arg_list_sac)

