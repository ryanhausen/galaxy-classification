import string

def format_params(dictionary):
    for k in dictionary.keys():
        if dictionary[k] == 'true':
            dictionary[k] = True
        elif dictionary[k] == 'false':
            dictionary[k] = False
        elif dictionary[k] == '':
            dictionary[k] = None
        elif type(dictionary[k]) not in  (float, int) and dictionary[k][0] == '[':
            # break up the list
            vals = dictionary[k][1:-1]
            vals = [v.strip() for v in vals.split(',')]

            # figure out what type is in the list
            val  = vals[0]
            is_num = False
            is_float = False

            for v in val:
                if v in string.digits:
                    is_num = True
                elif v == '.':
                    is_float = True

            # this is a list of numns
            if is_num:
                if is_float:
                    vals = [float(v) for v in vals]
                else:
                    vals = [int(v) for v in vals]
            # this is a list of strings
            else:
                vals = [str(v) for v in vals]

            dictionary[k] = vals
        elif type(dictionary[k]) == str:
            is_num = False
            is_float = False
            for s in dictionary[k]:
                if s in string.ascii_letters:
                    is_num = False
                    is_float = False
                    break
                elif s in string.digits:
                    is_num = True
                elif s == '.':
                    is_float = True

            if is_float:
                dictionary[k] = float(dictionary[k])
            elif is_num:
                dictionary[k] = int(dictionary[k])

    return dictionary
