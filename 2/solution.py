from itertools import permutations

##################
# COMMON STRINGS #
##################

START_OF_STRING = '^'
END_OF_STRING = '$'

######################
# PARENTHESIS REGEXP #
######################

LEFT_PARENTHESIS = [r'\(', r'\{', r'\[']
RIGHT_PARENTHESIS = [r'\)', r'\}', r'\]']
MAX_DEPTH = 7


def par_expr(depth: int):
    if depth == 1:
        prev = ""
    else:
        prev = par_expr(depth - 1)
    return fr"({'|'.join(l + prev + r for l, r in zip(LEFT_PARENTHESIS, RIGHT_PARENTHESIS))})*"


PARENTHESIS_REGEXP = START_OF_STRING + par_expr(MAX_DEPTH) + END_OF_STRING

####################
# SENTENCES REGEXP #
####################

END_OF_SENTENCE = [r'(\.)', r'(\!)', r'(\?)', r'(\.\.\.)', r'(\Z)']
LEFT_QUOTES = [r'\"', r"'", r'\«', r'\(']
RIGHT_QUOTES = [r'\"', r"'", r'\»', r'\)']
WORD_POSTFIXES = [',', ':', ';']
SPECIAL_FIRST_WORDS = [r'(Т\.к\.)', r'(Т\.о\.)', r'(Т\.д\.)', r'(Т\.е\.)', r'(P\.S\.)', r'(P\.s\.)']
SPECIAL_WORDS = [r'(—)', r'(т\.к\.)', r'(т\.о\.)', r'(т\.д\.)', r'(т\.е\.)']

FIRST_WORD_CHAR_GROUP = r'[ЁА-Я0-9]'
WORD_CHAR_GROUP = r'[ЁА-Яёа-я0-9\-]'
NON_QUOTES_GROUP = fr"[{'^' + ''.join(LEFT_QUOTES)}{''.join(RIGHT_QUOTES)}]"

QUOTES_WORD_ES = ['(' + le + NON_QUOTES_GROUP + '*' + ri + ')' for le, ri in zip(LEFT_QUOTES, RIGHT_QUOTES)]
QUOTED_FIRST_WORD_ES = ['(' + le + FIRST_WORD_CHAR_GROUP + NON_QUOTES_GROUP + '*' + ri + ')' for le, ri in zip(LEFT_QUOTES, RIGHT_QUOTES)]
FIRST_WORD_E = fr"({FIRST_WORD_CHAR_GROUP}{WORD_CHAR_GROUP}*)"
WORD_E = fr"({WORD_CHAR_GROUP}+)"
ENUM_ES = [r'(\d+\))', r'(\d+\.)']

FIRST_WORD_RE = fr"(({'|'.join(QUOTED_FIRST_WORD_ES + [FIRST_WORD_E] + SPECIAL_FIRST_WORDS)})[{''.join(WORD_POSTFIXES)}]?)"
WORD_RE = fr"(({'|'.join(QUOTES_WORD_ES + [WORD_E] + SPECIAL_WORDS)})[{''.join(WORD_POSTFIXES)}]?)"
ENUM_RE = fr"({'|'.join(ENUM_ES)})"

NORMAL_SENTENCE_RE = fr"({FIRST_WORD_RE}(\s+{WORD_RE})*({'|'.join(END_OF_SENTENCE)}))"
FULL_LIST_SENTENCE_RE = fr"({FIRST_WORD_RE}(\s+{WORD_RE})*\s*:(\s*{ENUM_RE}(\s+{WORD_RE})+\s*;)*(\s*{ENUM_RE}(\s+{WORD_RE})+\s*[.!]))"
ENUM_SENTENCE_RE = fr"({ENUM_RE}(\s+{WORD_RE})+({'|'.join(END_OF_SENTENCE)}))"
HEAD_LIST_SENTENCE_RE = fr"({FIRST_WORD_RE}(\s+{WORD_RE})*\s*:(?=(\s*{ENUM_RE})))"

SENTENCES_REGEXP = fr"(?P<sentence>({FULL_LIST_SENTENCE_RE}|{HEAD_LIST_SENTENCE_RE}|{ENUM_SENTENCE_RE}|{NORMAL_SENTENCE_RE}))"

##################
# PERSONS REGEXP #
##################

PERSONS_REGEXP = r'(?P<person>\b[А-Я][а-я]+ ([А-Я][а-я]+)+\b)'

#################
# SERIES REGEXP #
#################


def subpart_permutations(pieces):
    perms = permutations(pieces)
    return '(' + '|'.join(fr"({' '.join(perm)})" for perm in perms) + ')'


def concat_parts(parts):
    return r'\s*'.join(parts)


HTML_NAME_PARTS = [
    r'<h1.*>',
        fr'<a.*href=\"/series/\d+/\">',
            r'(?P<name>.*)',
        r'</a>',
    r'</h1>'
]
HTML_EPISODES_COUNT_PARTS = [
        r'<td.*>',
            r'<b>',
                r'Эпизоды:',
            r'</b>',
        r'</td>',
                    r'<td.*>',
                        r'(?P<episodes_count>\d+)',
                    r'</td>'
]
HTML_EPISODE_BODY_PARTS = [
            r'<span.*>',
                r'Эпизод (?P<episode_number>\d+)',
            r'</span>',
            r'<br/>',
            fr'<h1.*>',
                r'<b>',
                    r'(?P<episode_name>(.*))',
                r'</b>',
            r'</h1>',
            r'(<span.*>(?P<episode_original_name>.*)</span>\s*</td>)?',
            r'(<td.*>)?'
        fr'<td.*>(?P<episode_date>.*)</td>'
]
HTML_SEASON_BODY_PARTS = [
            fr'<td.*>',
                r'Сезон (?P<season>\d+)',
            r'</h1>',
            r'(?P<season_year>\d+), эпизодов: (?P<season_episodes>\d+)',
            r'</td>'
]

SERIES_BLOCKS = [
    HTML_NAME_PARTS,
    HTML_EPISODES_COUNT_PARTS,
    HTML_EPISODE_BODY_PARTS,
    HTML_SEASON_BODY_PARTS
]

SERIES_REGEXP = fr'{"|".join(concat_parts(block) for block in SERIES_BLOCKS)}'


if __name__ == "__main__":
    pass

    # print(len(PARENTHESIS_REGEXP))
    # print(PARENTHESIS_REGEXP)

    # print(FIRST_WORD_RE)
    # print(WORD_RE)
    # print(NORMAL_SENTENCE_RE)
    # print(FULL_LIST_SENTENCE_RE)
    # print(ENUM_SENTENCE_RE)
    # print(HEAD_LIST_SENTENCE_RE)
    # print()
    # print(SENTENCES_REGEXP)

    # print(concat_parts(HTML_NAME_PARTS))
    # print(concat_parts(HTML_EPISODES_COUNT_PARTS))
    # print(concat_parts(HTML_EPISODE_BODY_PARTS))
    # print(concat_parts(HTML_SEASON_BODY_PARTS))
    # print()
    # print(SERIES_REGEXP)

    '''
    
    import re

    regexp = re.compile(SERIES_REGEXP)
    html = ''.join(open("series/77164.html", "r").readlines())

    entities = set()
    for match in regexp.finditer(html):
        for key, value in match.groupdict().items():
            if value is not None:
                start, end = match.span(key)
                entities.add((start, end, key))

    for entity in entities:
        print(f'"{html[entity[0]:entity[1]]}"', f'({entity[2]})')
    
    '''
