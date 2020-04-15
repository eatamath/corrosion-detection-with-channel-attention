def LOG(*kargs):
#     global LOG_LEVEL
    if LOG_LEVEL == 'LOG':
        s = ''
        for arg in kargs:
            s += str(arg)
        s = 'DEBUG:: ' + s + '\n'
        print(s)
    return

