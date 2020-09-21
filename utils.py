import re

def get_first_token(suggestion):
    delimiters = [' ',
                  '_',
                  '.',
                  '(',
                  ')',
                  '{',
                  '}',
                  '[',
                  ']',
                  ',',
                  ':',
                  '\'',
                  '"',
                  '=',
                  '<',
                  '>',
                  '/',
                  '\\',
                  '+',
                  '-',
                  '|',
                  '&',
                  '*',
                  '%',
                  '=',
                  '$',
                  '#',
                  '@',
                  '!']
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, suggestion)[0]