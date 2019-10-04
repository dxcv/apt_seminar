import wrds
import inspect


all_functions = inspect.getmembers(wrds, inspect.isfunction)

print(all_functions)

db = wrds.Connection()
# db.create_pgpass_file()

# #%%
