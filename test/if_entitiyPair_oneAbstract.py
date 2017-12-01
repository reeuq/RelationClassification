with open('./../resource/1.1.relations.txt','r') as f:
    stringList = f.readlines()
    for string_wyd in stringList:
        tab1_string = string_wyd[string_wyd.find('(') + 1: string_wyd.find('.')]
        tab2_string = string_wyd[string_wyd.find(',') + 1: string_wyd.find('.', string_wyd.find('.') + 1)]
        if tab1_string != tab2_string:
            print(tab1_string)
            print(tab2_string)
            print("--------------------------------")