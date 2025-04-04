# Creation of a dictionary
my_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
# Printing the dictionary
print("Original dictionary:", my_dict)

# Indexing: Accessing a value using a key
value_b = my_dict['b']
print("Value at key 'b':", value_b)

# Slicing: Extracting a subset of the dictionary (using dictionary comprehension)
sliced_dict = {k: my_dict[k] for k in list(my_dict.keys())[1:3]}
print("Sliced dictionary:", sliced_dict)

# Concatenation: Merging two dictionaries
another_dict = {'e': 5, 'f': 6}
concatenated_dict = {**my_dict, **another_dict}
print("Concatenated dictionary:", concatenated_dict)

# Repetition: Not directly supported, but you can repeat keys/values in a list
repeated_dict = {k * 2: v * 2 for k, v in my_dict.items()}
print("Repeated dictionary:", repeated_dict)