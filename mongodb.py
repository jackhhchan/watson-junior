import os
import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["wiki"]         # create database

mycol = mydb["wiki"]           # create collection

# loop through files and create dictionary 
# with format -- {page_id : {passage_idx: passage, passage_idx: passage, ...}}
wiki_files = os.listdir('resource')
for wiki_file in wiki_files:
    path = "{}/{}".format('resource', wiki_file)
    



x = mycol.insert_one()

print(mydb.list_collection_names())


# x = mycol.find_one()

# print(x)