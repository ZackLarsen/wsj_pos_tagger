# Create the parser
my_parser = argparse.ArgumentParser(prog='tagger',
                                    description='Tag observations with part-of-speech (POS) tags.',
                                    add_help=False)

my_parser.add_argument('-a', action='store', default='42')
my_parser.add_argument('-a', action='store', type=int)

# Add AT LEAST ONE value:
my_parser.add_argument('input', action='store', nargs='+')

# FLEXIBLE number of values and store them in a list:
my_parser.add_argument('input',
                       action='store',
                       nargs='*',
                       default='my default value')

my_parser.version = '1.0'
my_parser.add_argument('-a', action='store')
my_parser.add_argument('-b', action='store_const', const=42)
my_parser.add_argument('-c', action='store_true')
my_parser.add_argument('-d', action='store_false')
my_parser.add_argument('-e', action='append')
my_parser.add_argument('-f', action='append_const', const=42)
my_parser.add_argument('-g', action='count')
my_parser.add_argument('-h', action='help')
my_parser.add_argument('-j', action='version')

# Set domain of allowed values:
my_parser.add_argument('-a', action='store', choices=['head', 'tail'])

# Make argument required:
my_parser.add_argument('-a',
                       action='store',
                       choices=['head', 'tail'],
                       required=True)

# Create mutually exclusive group of arguments that can't be entered at
# the same command:
my_group = my_parser.add_mutually_exclusive_group(required=True)
my_group.add_argument('-v', '--verbose', action='store_true')
my_group.add_argument('-s', '--silent', action='store_true')

# Specify a name for the value of an argument (using metavar keyword):
my_parser.add_argument('-v',
                       '--verbosity',
                       action='store',
                       type=int,
                       metavar='LEVEL')


args = my_parser.parse_args()

print(vars(args))
