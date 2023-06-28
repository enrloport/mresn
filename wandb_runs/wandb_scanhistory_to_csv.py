import pandas as pd
import wandb

run_path = "/elortiz/mnist_singlelayer__grid_search__ESN/t0ujyey3"
filename = "./csvs/mnist_singlelayer__grid_search__ESN__(500,6000).csv"

api  = wandb.Api()
run  = api.run(run_path)
hist = run.scan_history()
df   = pd.DataFrame.from_dict(hist)

df.to_csv(filename)