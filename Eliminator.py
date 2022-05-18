
import pickle


pickle_file = 'C:\\data_for_learning\\partial_val_data.pkl' ### Write path for the full_shape_val_data.pkl file ###

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)


objectlist=[]
for i in range(len(data)):
    if data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle'  or data[i]['semantic class'] == 'Bowl'  :
        objectlist.append(data[i])


with open(pickle_file, 'wb') as f:
    pickle.dump(objectlist, f, protocol=pickle.HIGHEST_PROTOCOL)


print(len(objectlist))