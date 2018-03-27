from preprocessing import read_data
import states
def main(data):
	states.run([i for i in data if len(i) > 0], 10, 2)
instances, labels = read_data('../data/Tweets.csv')
for l in set(labels):
	print "_"*50
	print l
	print "_"*50
	main([i for i, j in zip(instances, labels) if j == l])
	print ""
