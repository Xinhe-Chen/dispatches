import copy

generator_name = "303_WIND_1"
pem_name = generator_name + "_PEM"
installed_capacity = 847.0

def update_function(model_data, PEM_data):
    gen = model_data.data["elements"]["generator"][generator_name]

    pem = copy.deepcopy(gen)

    pem["p_min"] = 0.0
    pem["p_max"]["values"] = [min(PEM_data["PEM_power_capacity"], val) for val in gen["p_max"]["values"] ]
    pem["p_cost"] = PEM_data["PEM_bid"]

    model_data.data["elements"]["generator"][pem_name] = pem
    for idx, val in enumerate(gen["p_max"]["values"]):
        gen["p_max"]["values"][idx] = max(0., val - PEM_data["PEM_power_capacity"])