from random import randint

def run(statements, samples=10, prev=1):
	validletters = '1234567890abcdefghijklmnopqrstuvwxyz,. \n:?'
	statements = ["$ "*prev + ("".join([letter for letter in statement.lower() if letter in validletters])) + " /" for statement in statements]
	originals = [s[2*prev:-2] for s in statements]
	statement = statement
	words = {}
	for s in statements:
		w = s.split(" ")
		for i in range(prev, len(w)):
			try:
				words[" ".join(w[i-prev:i])].append(w[i])
			except:
				words[" ".join(w[i-prev:i])] = [w[i]]
	for aeioja in range(samples):
		q = ["$"] * prev
		text = ""
		while True:
			arr = words[" ".join(q)]
			word = arr[randint(0, len(arr)-1)]
			if word == "/":
				if text[1:] in originals:
					q = ["$"] * prev
					text = ""
				else:
					print '"'+text[1:]+'"'
					break
			else:
				q = q[1:] + [word]
				text = text + " " + word
