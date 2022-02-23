#This script plots the surrogate results for a single rankine cycle solution. 
#It creates a bar chart for the number of hours in each operating zone.
import json, os
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)

for file in os.listdir('rankine_results/scikit_surrogate'):
    if 'json' in file:

        with open('rankine_results/scikit_surrogate/{}'.format(file)) as f:
            data = json.load(f)

        #read the surrogate solution
        x = data["market_inputs"]
        dispatch_zones = data["scaled_dispatch_zones"]

        fig, ax = plt.subplots(figsize = (16,8))
        ax.set_xlabel("Scaled Power Output (% of maximum)", fontsize=24)
        ax.set_xticks(range(len(dispatch_zones)))
        ax.tick_params(axis='x', labelrotation = 45)
        ax.set_xticklabels(["Off","0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"])

        ax.bar(range(len(dispatch_zones)),dispatch_zones, color="blue")
        ax.set_ylabel("Hours in Operating Zone", fontsize=24)
        plt.tight_layout()

        name = file.split('.')[0]
        fig.savefig("rankine_results/scikit_surrogate/{}".format(name))