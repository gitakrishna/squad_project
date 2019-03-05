import numpy as np
import random 
import json 


def main():
	with open('./data/train-v2.0.json') as json_file: 
		file = json.load(json_file)
		# print(data)
		count = 0
		# print(len(file))
		# print(type(file))
		data = file['data']
		all_dicts = data[0]

		small_train = {}
		small_train['version'] = 'v2.0'


		small_data = []

		num_questions = 0


		# print(data)
		for line in file['data']:
			x = random.randint(1, 4)
			if x == 1: 
			# print("line ", count, ": ", line)
				# print(line['paragraphs'])
				paras = line['paragraphs']
				# print("PARAS: ", len(paras))
				# print("paras: ", paras)
				small_paras = []
				# print("len paras: ", len(paras))
				for thing in paras: 
					y = random.randint(1, 7)
					if y == 1:
						small_paras.append(thing)
						num_questions += len(thing['qas'])
				tiny = {}
				tiny['paragraphs'] = small_paras
				tiny['title'] = line['title']
				small_data.append(tiny)
				# small_data['paragraphs'] = small_paras
				# small_data['title'] = line['title']
				# print(line.keys())
				# print()
					# print("thing: ", thing)
					# print(thing.keys())
					# print("thing length: ", len(thing))
					# break
					# qas = thing['qas']
					# small_qas = []
					# small_thing = {}
					# # print("thing: ", thing)
					# print("thing type: ", type(thing))
					# print("qas type: ", type(qas))
					# for qa in qas:
					# 	print ("question and answer: ", qa)

						# break
		# print(small_data)
		print("size: ", len(small_data))
		print("questions: ", num_questions)



		small_train['data'] = small_data

		print(len(file))

		with open('train_small.json', 'w') as outfile: 
			json.dump(small_train, outfile)

if __name__ == '__main__':
	main()
