##################
# COMMON STRINGS #
##################

START_OF_STRING = '^'
END_OF_STRING = '$'

####################
# PASSWORD REGEXP #
####################

SPEC_SYMS = r'$^%@#&*!?'
CONTAINS_DIGIT = r'(?=.*[0-9])'
CONTAINS_LOWERCASE_LATIN = r'(?=.*[a-z])'
CONTAINS_UPPERCASE_LATIN = r'(?=.*[A-Z])'
CONTAINS_TWO_SPEC_SYMS = fr'(?=.*(?P<special>[{SPEC_SYMS}]).*((?!(?P=special))[{SPEC_SYMS}]))'
CONTAINS_NOT_TWO_IDENTICAL_CONSECUTIVE_CHARS = r'(?!.*(?P<symbol>.)(?P=symbol))'
PASSWORD_BASE = fr'[0-9a-zA-Z{SPEC_SYMS}]{{8,}}'

PASSWORD_REGEXP = START_OF_STRING + \
                  CONTAINS_DIGIT + CONTAINS_TWO_SPEC_SYMS + \
                  CONTAINS_LOWERCASE_LATIN + CONTAINS_UPPERCASE_LATIN + \
                  CONTAINS_NOT_TWO_IDENTICAL_CONSECUTIVE_CHARS + \
                  PASSWORD_BASE + \
                  END_OF_STRING

################
# COLOR REGEXP #
################


def gen_range_re(start, end):
    return '(' + '|'.join(map(str, range(start, end))) + ')'


def three_component_bracket(c1, c2, c3):
    return fr'\([ \t]*{c1}[ \t]*,[ \t]*{c2}[ \t]*,[ \t]*{c3}[ \t]*\)'


RANGE_0_255 = gen_range_re(0, 256)
RANGE_0_100 = gen_range_re(0, 101)
RANGE_0_360 = gen_range_re(0, 361)
HEX_DIGIT = '[0-9a-fA-F]'

HEX_LONG_REGEXP = fr'#{HEX_DIGIT}{{6}}'
RGB_NUMS_REGEXP = 'rgb' + three_component_bracket(RANGE_0_255, RANGE_0_255, RANGE_0_255)
RGB_PERC_REGEXP = 'rgb' + three_component_bracket(RANGE_0_100 + '%', RANGE_0_100 + '%', RANGE_0_100 + '%')
HEX_SHORT_REGEXP = fr'#{HEX_DIGIT}{{3}}'
HSL_REGEXP = 'hsl' + three_component_bracket(RANGE_0_360, RANGE_0_100 + '%', RANGE_0_100 + '%')

COLOR_REGEXP = START_OF_STRING + \
               fr'(({HEX_SHORT_REGEXP})|({HEX_LONG_REGEXP})|({RGB_NUMS_REGEXP})|({RGB_PERC_REGEXP})|({HSL_REGEXP}))' + \
               END_OF_STRING

#####################
# EXPRESSION_REGEXP #
#####################

VAR_FIRST_SYMS = 'a-zA-Z_'
VAR_SYMS = VAR_FIRST_SYMS + '0-9'
CONSTANTS = ['pi', 'e', 'sqrt2', 'ln2', 'ln10']
FUNCTIONS = ['sin', 'cos', 'tg', 'ctg', 'tan',
             'cot', 'sinh', 'cosh', 'th', 'cth',
             'tanh', 'coth', 'ln', 'lg', 'log',
             'exp', 'sqrt', 'cbrt', 'abs', 'sign']
OPERATORS = ['*', '/', r'\-', '+', '^']


VARIABLE_REGEXP = fr'(?P<variable>[{VAR_FIRST_SYMS}][{VAR_SYMS}]*)'
NUMBER_REGEXP = r'(?P<number>\d+(\.\d*)?)'
CONSTANT_REGEXP = fr'(?P<constant>\b(' + '|'.join(CONSTANTS) + r')\b)'
FUNCTION_REGEXP = fr'(?P<function>\b(' + '|'.join(FUNCTIONS) + r')\b)'
OPERATORS_REGEXP = r'(?P<operator>[' + ''.join(OPERATORS) + '])'
LEFT_PARENTHESIS_REGEXP = r'(?P<left_parenthesis>\()'
RIGHT_PARENTHESIS_REGEXP = r'(?P<right_parenthesis>\))'
EXPRESSION_PARTS = [FUNCTION_REGEXP, CONSTANT_REGEXP,
                    NUMBER_REGEXP, VARIABLE_REGEXP, OPERATORS_REGEXP,
                    LEFT_PARENTHESIS_REGEXP, RIGHT_PARENTHESIS_REGEXP]


EXPRESSION_REGEXP = '|'.join(EXPRESSION_PARTS)

################
# DATES REGEXP #
################

MOTHS_RU = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
MOTHS_ENG = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
MOTHS_ENG_SHORT = [m[:3] for m in MOTHS_ENG]

YEAR_REGEXP = r'(\d+)'

MONTH_NUM_REGEXP = r'((0?[1-9])|([1-2]\d)|(3[01]))'
MONTH_RU_REGEXP = "(" + "|".join(MOTHS_RU) + ")"
MONTH_ENG_REGEXP = "(" + "|".join(MOTHS_ENG) + ")"
MONTH_ENG_SHORT_REGEXP = "(" + "|".join(MOTHS_ENG_SHORT) + ")"

DAY_REGEXP = r'((0?[1-9])|([1-2]\d)|(3[01]))'

DATES_ALTERNATIVES = [
    fr'{DAY_REGEXP}\.{MONTH_NUM_REGEXP}\.{YEAR_REGEXP}',
    fr'{DAY_REGEXP}/{MONTH_NUM_REGEXP}/{YEAR_REGEXP}',
    fr'{DAY_REGEXP}-{MONTH_NUM_REGEXP}-{YEAR_REGEXP}',
    fr'{YEAR_REGEXP}\.{MONTH_NUM_REGEXP}\.{DAY_REGEXP}',
    fr'{YEAR_REGEXP}/{MONTH_NUM_REGEXP}/{DAY_REGEXP}',
    fr'{YEAR_REGEXP}-{MONTH_NUM_REGEXP}-{DAY_REGEXP}',
    fr'{DAY_REGEXP} {MONTH_RU_REGEXP} {YEAR_REGEXP}',
    fr'{MONTH_ENG_REGEXP} {DAY_REGEXP}, {YEAR_REGEXP}',
    fr'{MONTH_ENG_SHORT_REGEXP} {DAY_REGEXP}, {YEAR_REGEXP}',
    fr'{YEAR_REGEXP}, {MONTH_ENG_REGEXP} {DAY_REGEXP}',
    fr'{YEAR_REGEXP}, {MONTH_ENG_SHORT_REGEXP} {DAY_REGEXP}'
]

DATES_REGEXP = START_OF_STRING + \
               '|'.join(f'({alt})' for alt in DATES_ALTERNATIVES) + \
               END_OF_STRING

'''
print(PASSWORD_REGEXP)
print(COLOR_REGEXP)
print(EXPRESSION_REGEXP)
print(DATES_REGEXP)
'''

'''
if __name__ == "__main__":
    import re

    if False:
        s = "epi2"
        pattern = EXPRESSION_REGEXP
        print(s)
        for match in re.finditer(pattern=pattern, string=s):
            print(f'type: {match.lastgroup}, span: {match.span()}')
'''
