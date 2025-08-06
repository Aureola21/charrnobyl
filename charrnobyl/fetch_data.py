#get the name of the input file
print("Asssumption:")
print("---> The data is stored in a file, under the folder data")
print("---> The file should contain one word per line")
print()
file_name = input('Enter the name of the input file (e.g., indian_cities.txt): ')
file_name = 'data/' + file_name

# Loading the data
input_data= open(file_name, 'r').read().splitlines()
input_data = [w.lower() for w in input_data]

