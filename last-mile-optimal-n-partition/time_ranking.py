import os
import matplotlib.pyplot as plt

# Specify the directory containing text files
directory = './timerecords_plus'

# Initialize an empty dictionary to store the sum of each file
sums = {}

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            total = 0
            for line in file:
                try:
                    # Assuming each line in the file contains a number
                    total += float(line.strip())
                except ValueError:
                    # Handle the case where the line is not a number
                    pass
            sums[filename] = total

# Sort the sums in decreasing order
sorted_sums = dict(sorted(sums.items(), key=lambda item: item[1], reverse=True))

# Plotting
plt.bar(sorted_sums.keys(), sorted_sums.values())
plt.xticks(rotation=45, ha='right')
plt.ylabel('Summation')
plt.yscale('log')
plt.title('Sum of Data in Text Files')
plt.tight_layout()
plt.show()
# plt.savefig('times.png')
