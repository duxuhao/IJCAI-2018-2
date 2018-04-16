import subprocess

with open("para.log",'w') as f:
    f.write("depth,leaves,learn,logloss\n")
for md in range(3,6):
    for nl in range(6,20,3):
        for lr in [0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03]:
            x = subprocess.run(["./lightgbm", "config=TRAIN.conf", "max_depth={}".format(md), "num_leaves={}".format(nl), "learning_rate={}".format(lr)], stdout=subprocess.PIPE)
            with open("para.log",'a') as f:
                try:
                    f.write("{},{},{},{}\n".format(md, nl, lr, float(str(x.stdout).split('\\n')[-5].split(':')[-1])))
                except:
                   pass
