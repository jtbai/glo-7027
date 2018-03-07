from pickle import load


the_data = load(open('./prepared_data.pyk', 'rb'))
print(the_data)