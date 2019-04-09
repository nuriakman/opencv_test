#Kaynak: https://stackoverflow.com/questions/8884188/how-to-read-and-write-ini-file-with-python3
from configparser import ConfigParser

# instantiate
config = ConfigParser()

# parse existing file
config.read('test.ini')

# read values from a section
string_val = config.get('section_a', 'string_val')
bool_val = config.getboolean('section_a', 'bool_val')
int_val = config.getint('section_a', 'int_val')
float_val = config.getfloat('section_a', 'pi_val')

# update existing value
config.set('section_a', 'string_val', 'world')

# add a new section and some values
config.add_section('section_b')
config.set('section_b', 'meal_val', 'spam')
config.set('section_b', 'not_found_val', 404)

# save to a file
with open('test_update.ini', 'w') as configfile:
    config.write(configfile)
