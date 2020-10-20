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


def escape_spaces_from_beggining(suggestion):
  matrix = suggestion.split(' ')
 
  while matrix !=[] and not matrix[0]:
    del matrix[0]

  return ' '.join(matrix)