import subprocess
from time import sleep
import math

# for simple commands
# subprocess.run(["ls", "-l"]) 

# for complex commands, with many args, use string + `shell=True`:
duration = 5
old_losses = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
new_losses = [0.08 + i * 0.01 for i in range(14)]

l = old_losses + new_losses
for fec in ["fec_dummy", "fec_1pr", "fec_2pr", "fec_r"]:
    for loss in l:
        for fec_batch_size in [8, 16, 32, 64]:
            if fec == "fec_dummy":
                rate = 10
            elif fec == "fec_1pr":
                rate = 10 * fec_batch_size / (fec_batch_size + 1)
            elif fec == "fec_2pr":
                rate = 10 * fec_batch_size / (fec_batch_size + 2)
            elif fec == "fec_r":
                rate = 10 * fec_batch_size / (fec_batch_size + 2 + int(math.log2(fec_batch_size)))
            else:
                rate = 0
                raise(Exception("wrong fec"))
            
            cmd_str = f"sudo python3 script.py --test --loss {loss} --fec {fec} --fec_batch_size {fec_batch_size} --rate {rate}"

            for i in range(10):
                subprocess.run(cmd_str, shell=True)

                # sleep(duration + 2)
                sleep(1)
                subprocess.run(f"cp iperf_srv_log2.txt data/{fec}/srv_{fec_batch_size}_{loss}_{duration}_1{i}.txt", shell=True)    
                subprocess.run(f"cp iperf_cli_log1.txt data/{fec}/cli_{fec_batch_size}_{loss}_{duration}_1{i}.txt", shell=True)    
