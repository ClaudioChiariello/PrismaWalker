
import pickle
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


weight_path = "/home/claudio/raisim_ws/raisimlib/raisimGymTorch/data/prisma_walker_locomotion/lam_90/full_300.pt"


iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

lam = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
plt.figure(1)
for lam in lam:
    with open(weight_dir +  "../lam_" + str(int(100*lam)) + "/loss/loss.pkl", 'rb') as fp:
        loss = pickle.load(fp)
        
    loss = loss.values()
    loss = list(loss)

    episode = range(len(loss))
    plt.plot(episode, loss, label="lambda ="+str(lam))
    plt.grid(linestyle='-', linewidth=0.8)
    plt.xlabel('Episodes')
    plt.ylabel("Loss function")
    plt.ylim([-0.5,0.9])
    plt.legend()

plt.show()  #Ogni volta che invochi plt.plot prima di show, aggiungi un disegno allo stesso grafico
