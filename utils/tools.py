import json

def load_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        jsonfile = f.read().encode('utf-8')
        index_map = json.loads(jsonfile)
        return index_map


def write_json(index_map, path):
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(index_map, f, sort_keys=False, indent=8, ensure_ascii=True)


def write_list(info_list, path):
    with open(path, 'w') as f:
        for info in info_list:
            str_info = [str(i) for i in info]
            s = ' '.join(str_info) + '\n'
            f.write(s)

def load_list(path):
    list_needed = []
    with open(path, 'r') as f:
        for line in f:
            if line == '\n':
                list_needed.append([])
            else:
                str_info = line.split(" ")
                info = [int(index) for index in str_info]
                list_needed.append(info)
    return list_needed