import os
import subprocess
import time
configurations = [c for c in os.listdir("./configurations/grid_search") if c.endswith(".yaml") and c.startswith("configuration")]
for config in configurations:
	command = ["sbatch", "./jobs/train.sbatch", os.path.join("./configurations/grid_search", config)]
	print(command)
	subprocess.call(command)
	time.sleep(1)
