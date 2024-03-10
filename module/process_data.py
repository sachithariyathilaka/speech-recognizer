import csv
import json
import random


def count(csv_path):
    with open(csv_path) as f:
        return sum(1 for line in f)


def generateData(csv_path, audio_path):
    length = count(csv_path)
    data = []

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        index = 1
        print(str(length) + " files found")

        for row in reader:
            file_path = row['path']
            text = row['sentence']
            data.append({
                "key": audio_path + file_path,
                "text": text
            })
            index = index + 1

        return data


def generateJson(data, json_path, percent):
    random.shuffle(data)

    print("Generating the train.json")
    with open(json_path + "/" + 'train.json', 'w') as f:
        d = len(data)
        i = 0

        while i < int(d - d / percent):
            r = data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i + 1
    print("Generated the train.json")

    print("Generating the test.json")
    with open(json_path + "/" + 'test.json', 'w') as f:
        d = len(data)
        i = int(d - d / percent)

        while i < d:
            r = data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i + 1
    print("Generated the test.json")


class ProcessData:

    def __init__(self, csv_path, audio_path, json_path, percent):
        data = generateData(csv_path, audio_path)
        generateJson(data, json_path, percent)


if __name__ == "__main__":
    csv_path = 'D:/dataset/en/validated.tsv'
    audio_path = 'D:/dataset/en/clips/'
    json_path = "D:/dataset/"
    percent = 10
    ProcessData(csv_path, audio_path, json_path, percent)
