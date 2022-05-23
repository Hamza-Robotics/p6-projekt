
import pickle


pickle_file = 'D:\\data_for_learning\\partial_train_data.pkl' ### Write path for the full_shape_val_data.pkl file ###

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

print(len(data))
objectlist=[]
for i in range(len(data)):
    if data[i]['semantic class'] == 'Knife' or data[i]['semantic class'] == 'Bottle'  or data[i]['semantic class'] == 'Bowl' or data[i]['semantic class'] == 'Mug'  :
        objectlist.append(data[i])


with open(pickle_file, 'wb') as f:
    pickle.dump(objectlist, f, protocol=pickle.HIGHEST_PROTOCOL)


print(len(objectlist))