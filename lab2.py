# Str ops
s1 = "Python"
s2 = "Programming"

# Indexing
print("Indexing s1[0]:", s1[0])
# Slicing
print("Slicing s1[1:4]:", s1[1:4])
# Concat
s_concat = s1 + " " + s2
print("Concatenated Str:", s_concat)
# Repeat
s_repeat = s1 * 2
print("Repeated Str:", s_repeat)

# List ops
l1 = [7, 8, 9]
l2 = [10, 11, 12]

# Indexing
print("Indexing l1[0]:", l1[0])
# Slicing
print("Slicing l1[1:3]:", l1[1:3])
# Concat
l_concat = l1 + l2
print("Concatenated List:", l_concat)
# Repeat
l_repeat = l1 * 2
print("Repeated List:", l_repeat)

# Tuple ops
t1 = (7, 8, 9)
t2 = (10, 11, 12)

# Indexing
print("Indexing t1[0]:", t1[0])
# Slicing
print("Slicing t1[1:3]:", t1[1:3])
# Concat
t_concat = t1 + t2
print("Concatenated Tuple:", t_concat)
# Repeat
t_repeat = t1 * 2
print("Repeated Tuple:", t_repeat)

# Set ops
s1 = {6, 7, 8}
s2 = {8, 9, 10}

# Union
s_union = s1 | s2
print("Union Set:", s_union)
# Intersection
s_inter = s1 & s2
print("Intersection Set:", s_inter)

# Dict ops
d1 = {'e': 5, 'f': 6}
d2 = {'g': 7, 'h': 8}

# Indexing
print("Indexing d1['e']:", d1['e'])
# Merge
d_merge = {**d1, **d2}
print("Merged Dict:", d_merge)
