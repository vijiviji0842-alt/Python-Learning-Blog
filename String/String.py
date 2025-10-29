text = input("Enter a sentence: ")
words = text.split()
count = {}

for w in words:
    count[w] = count.get(w, 0) + 1

print(count)
