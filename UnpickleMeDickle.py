import pickle

pickle_file = 'C:\\Users\\SebBl\\source\\repos\\UnpickleMyDickle\\full_shape_val_data.pkl' ### Write path for the full_shape_val_data.pkl file ###

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
y = 0
for x in data[0]['full_shape']['coordinate']:
    print((data[0]['full_shape']['coordinate'][y]))
    y+=1