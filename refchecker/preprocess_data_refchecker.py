from data import get_dataset, extract_datapoint_info
import json
import os

hard_questions = ["potential of class activation mapping in cxr",
                  "how does the percentage of agricultural products used for fish feed production in europe compare to other regions?",
                  "what are some notable achievements or publications by roberta fusaro in her career?",
                  "what is the process of water sowing and how does it differ from traditional methods of irrigation?"]


def create_extractor_input(data, data_name, save_path):
    if not os.path.exists(save_path):

        input_list = []
        for idx, datapoint in data.iterrows():
            dp = extract_datapoint_info(datapoint, data_name)

            if data_name == "selfcheck":
                prompt = get_prompt(dp)
                sentece_level = True
                if sentece_level:
                    for sentence in dp.sentences:
                        dict_dp = {"dp_idx": idx, "response": sentence, "question": prompt}
                        input_list.append(dict_dp)
                else:
                    dict_dp = {"response": dp.gen_text, "question": prompt}
                    input_list.append(dict_dp)

            elif data_name == "scopus":

                question = data["query"][idx]
                # if dp.gen_text.startswith("None of the provided"):
                #     continue
                if question in hard_questions:
                    dict_dp = {"dp_idx": idx, "response": dp.gen_text, "question": question}
                    input_list.append(dict_dp)

        with open(save_path, "w") as f:
            json.dump(input_list, f)


def create_checker_input(data, data_name, save_path):
    if data_name == "selfcheck":
        path = "data/refchecker/refchecker_triplets.json"
    elif data_name == "scopus":
        path = "data/refchecker/refchecker_triplets_scopus_hard.json"
    with open(path, "r") as f:
        triplets_list = json.load(f)

    input_list = []

    for triplet in triplets_list:
        dp_idx = triplet["dp_idx"]
        dp = extract_datapoint_info(data.iloc[dp_idx], data_name)
        trip = triplet["triplets"]
        sentence = triplet["response"]

        if data_name == "selfcheck":
            dict_dp = {"dp_idx": dp_idx, "response": sentence, "triplets": trip, "reference": dp.ext_knowledge[0]}
        else:
            dict_dp = {"dp_idx": dp_idx, "response": sentence, "triplets": trip, "reference": dp.ext_knowledge}
        input_list.append(dict_dp)

    with open(save_path, "w") as f:
        json.dump(input_list, f)


def exact_match_count(triplet, sentence):
    count = 0
    for comp in triplet:
        if comp in sentence:
            count += 1
    return count


def get_best_match(triplets, sentences):
    """For each sentence, get the matching triplets"""

    best_match = {}
    for idx_t, triplet in enumerate(triplets):
        best_match[idx_t] = {"idx": 0, "count": 0}
        for idx_s, sentence in enumerate(sentences):
            count = exact_match_count(triplet, sentence)
            if count > best_match[idx_t]["count"]:
                best_match[idx_t]["count"] = count
                best_match[idx_t]["idx"] = idx_s

    # for each sentence, get the list of triplets that match it best
    sentence_triplets = {}
    for idx_t, match in best_match.items():
        if match["idx"] not in sentence_triplets:
            sentence_triplets[match["idx"]] = []
        sentence_triplets[match["idx"]].append(idx_t)

    return sentence_triplets


def get_prompt(dp):
    concept = ""
    mid_names = ["of", "de", "van", "von", "the", "The"]
    concept_list = dp.ext_knowledge[0].split(" ")
    for string in concept_list:

        if string in mid_names or string.isupper() or string[0].isupper():
            if string[0] != "(":
                concept += string + " "
        else:
            break

    return f"This is a Wikipedia passage about {concept.strip()}:"



def aggregate_results(data):
    with open("data/refchecker/refchecker_input_checker.json", "r") as f:
        input_list = json.load(f)

    with open("RESULTS", "r") as f:
        results = json.load(f)

    for idx, result in enumerate(results):
        dp = extract_datapoint_info(data.iloc[idx], data)
        triplets = input_list[idx]["triplets"]
        matches = get_best_match(triplets, dp.sentences)

        for idx, label in dp.annotation:
            trip_per_sentence = matches[idx]


data_name = "selfcheck"
save_path = "data/scopus/refchecker_input_extractor_scopus_hard.json"
save_path_checker = "data/scopus/refchecker_input_checker_scopus_hard.json"
dataset = get_dataset(data_name)
#
# create_extractor_input(dataset, data_name, save_path)

create_checker_input(dataset, data_name, save_path_checker)

