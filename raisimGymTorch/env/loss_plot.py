import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading the pickle file: {e}")
        return None

def print_loss(x, y ):

    plt.plot(x, label=y)

    plt.title('Loss function varying lambda')
    plt.xlabel('Episode')
    plt.ylabel('Loss value')
    plt.ylim(2, 50) 
    plt.legend()
    


# Example usage
if __name__ == "__main__":
    task_path = os.path.dirname(os.path.realpath(__file__))

    lam = [9, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    file_paths = []
    for i in lam:
        path_to_data = task_path + "/../../.." + "/raisimGymTorch/data/prisma_walker_locomotion/" 
        file_paths.append(path_to_data + str(i) + "/loss/loss.pkl")

        # Read the pickle file
        loss_dict = read_pickle_file(file_paths[-1])

        # Process the loaded data (replace this with your actual data processing logic)
        if loss_dict is not None:
            x_axis = loss_dict.values() #prendo solo i valori del dizionario
            x_axis = list(x_axis)

            print_loss(x_axis, "lambda = " + str(i) )

    plt.grid()
    plt.show()
