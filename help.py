a = [1,3,42,3,4]
b = [14,53,42,573,556]
import numpy
c = []
c.append(b)
c.append(a)
print(c)
c.sort()
print(c)
#
# import bisect
# a = [1,2,3,4,5]
# print(bisect.bisect_left(a,0.9))

# with open('names.csv', 'w') as csvfile:
#     fieldnames = ['first_name', 'last_name']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     writer.writeheader()
#     writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
#     writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
#     writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})

# csv.register_dialect('myDialect',
#                      delimiter=';',
#                      quoting=csv.QUOTE_ALL,
#                      skipinitialspace=False)
# with open('bank-full.csv') as csvfile:
# import csv
# with open('names.csv') as csvfile:
#     reader = csv.DictReader(csvfile, delimiter=',')
#     line=0
#     for row in reader:
#         print(row[0])
#         line = line+1
#     print(line)
# #         # print(row['age'])
# #     # for row in reader:
# #         print (row)
#
# w, h = 8, 5;
# Matrix = [[0 for x in range(w)] for y in range(h)]
# Matrix[0][0] = 1
# Matrix[6][0] = 3  # error! range...
# Matrix[0][6] = 3  # valid
#
# print
# Matrix[0][0]  # prints 1
# x, y = 0, 6
# print
# Matrix[x][y]  # prints 3; be careful with indexing!
#
# matrix = [[]]
#
# Matrix = {}
# Matrix[1, 2] = 15
# print( Matrix[1, 2])
#
# data = [[]]
# csv.register_dialect('myDialect',
#                      delimiter=';',
#                      quoting=csv.QUOTE_ALL,
#                      skipinitialspace=False)
#
# with open('bank-full.csv', 'r') as csvFile:
#     reader = csv.DictReader(csvFile, dialect='myDialect')
#     # reader = csv.Reader(f, dialect='myDialect')
#     for row in reader:
#         print(len(row))
#         print(row)
#         print(dict(row))
#         # print(row[2])
# csvFile.close()

# metaData = []
# lineno = 0
# with open('meta-data.txt', 'r') as f:
#     for line in f:
#         metaData.append([])
#         wordno = 0
#         for word in line.split():
#             metaData[lineno].append(word)
#             # print( word)
#         lineno = lineno + 1
#
# for i in range(len(metaData)):
#     for word in metaData[i]:
#         sys.stdout.write(word + " ")
#     print()