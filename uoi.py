import h5py

# Replace 'model.h5' with your actual .h5 file name
model_path = 'C:\\Users\\sujay\\Desktop\\nro\\models\\asl_model.h5'

with h5py.File(model_path, 'r') as model_file:
    def print_structure(name):
        print(name)
        
    # Print the structure of the HDF5 file
    model_file.visit(print_structure)
