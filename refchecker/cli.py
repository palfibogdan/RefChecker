import os
os.environ["HF_HOME"] = "/nlp/cache/huggingface/"
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from tqdm import tqdm

from refchecker.extractor import Claude2Extractor, GPT4Extractor, MixtralExtractor
from refchecker.checker import Claude2Checker, GPT4Checker, NLIChecker, AlignScoreChecker, ZephyrChecker
from refchecker.retriever import GoogleRetriever
from refchecker.aggregator import strict_agg, soft_agg, major_agg


def get_args():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "mode", nargs="?", choices=["extract", "check", "extract-check"],
        help="extract:       Extract triplets from provided responses.\n"
             "check:         Check whether the provided triplets are factual.\n"
             "extract-check: Extract triplets and check whether they are factual."
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Input path to the json file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output path to the result json file."
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./.cache",
        help="Path to the cache directory. Default: ./.cache"
    )
    parser.add_argument(
        '--extractor_name', type=str, default="claude2",
        choices=["gpt4", "claude2", "mixtral"],
        help="Model used for extracting triplets. Default: claude2."
    )
    parser.add_argument(
        '--extractor_max_new_tokens', type=int, default=500,
        help="Max generated tokens of the extractor, set a larger value for longer documents. Default: 500"
    )
    parser.add_argument(
        "--checker_name", type=str, default="claude2",
        choices=["gpt4", "claude2", "nli", "alignscore", "zephyr"],
        help="Model used for checking whether the triplets are factual. "
        "Default: claude2."
    )
    parser.add_argument(
        "--retriever_name", type=str, default="google", choices=["google"],
        help="Model used for retrieving reference (currently only google is"
        " supported). Default: google."
    )
    parser.add_argument(
        "--aggregator_name", type=str, default="soft",
        choices=["strict", "soft", "major"],
        help="Aggregator used for aggregating the results from multiple "
             "triplets. Default: soft.\n"
             "*  strict: If any of the triplets is Contradiction, the response"
             " is Contradiction.\nIf all of the triplets are Entailment, the "
             "response is Entailment. Otherwise, the\nresponse is Neutral.\n"
             "*  soft:   The ratio of each category is calculated.\n"
             "*  major:  The category with the most votes is selected."
    )
    parser.add_argument(
        "--openai_key", type=str, default="",
        help="Path to the openai api key file. Required if openAI models are"
        " used."
    )
    parser.add_argument(
        "--anthropic_key", type=str, default="",
        help="Path to the Anthropic api key file. Required if the Anthropic "
        "Claude2 api is used."
    )
    parser.add_argument(
        "--aws_bedrock_region", type=str, default="",
        help="AWS region where the Amazon Bedrock api is deployed. Required if "
        "the Amazon Bedrock api is used."
    )
    parser.add_argument(
        "--use_retrieval", action="store_true",
        help="Whether to use retrieval to find the reference for checking. "
        "Required if the reference\nfield in input data is not provided."
    )
    parser.add_argument(
        "--serper_api_key", type=str, default="",
        help="Path to the serper api key file. Required if the google retriever"
        " is used."
    )

    return parser.parse_args()


def main():
    args = get_args()
    # set environment variables
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://els-sdanswers-innovation.openai.azure.com/"

    if args.openai_key:
        with open(args.openai_key, "r") as fp:
            os.environ["OPENAI_API_KEY"] = fp.read().strip()
            os.environ["AZURE_OPENAI_API_KEY"] = fp.read().strip()

    if args.anthropic_key:
        with open(args.anthropic_key, "r") as fp:
            os.environ["ANTHROPIC_API_KEY"] = fp.read().strip()
    if args.aws_bedrock_region:
        os.environ["aws_bedrock_region"] = args.aws_bedrock_region
    if args.serper_api_key:
        os.environ["SERPER_API_KEY"] = args.serper_api_key

    if args.mode == "extract":
        extract(args)
    elif args.mode == "check":
        check(args)
    elif args.mode == "extract-check":
        output_path = args.output_path
        args.output_path = output_path + ".temp"
        extract(args)
        args.input_path = args.output_path
        args.output_path = output_path
        check(args)
    else:
        raise NotImplementedError


def extract(args):
    # initialize models
    if args.extractor_name == "claude2":
        extractor = Claude2Extractor()
    elif args.extractor_name == "gpt4":
        extractor = GPT4Extractor()
    elif args.extractor_name == "mixtral":
        extractor = MixtralExtractor()
    else:
        raise NotImplementedError

    # load data
    with open(args.input_path, "r") as fp:
        input_data = json.load(fp)
    
    # extract triplets
    print('Extracting')
    output_data = []

    existing_triplets = []
    existing_output = None
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as fp:
            existing_output = json.load(fp)
            existing_triplets = [t["response"] for t in existing_output]

    for idx, item in enumerate(tqdm(input_data)):
        assert "response" in item, "response field is required"
        response = item["response"]
        if response in existing_triplets:
            continue

        question = item.get("question", None)
        triplets = extractor.extract_claim_triplets(response, question, max_new_tokens=args.extractor_max_new_tokens)
        out_item = {**item, **{"triplets": triplets}}
        output_data.append(out_item)

        if idx % 100 == 0:
            with open(args.output_path, "w") as fp:
                out = output_data
                if existing_output is not None:
                    out = existing_output + output_data
                json.dump(out, fp, indent=2)

    with open(args.output_path, "w") as fp:
        out = output_data
        if existing_output is not None:
            out = existing_output + output_data
        json.dump(out, fp, indent=2)

def check(args):
    # initialize models
    if args.checker_name == "claude2":
        checker = Claude2Checker()
    elif args.checker_name == "gpt4":
        checker = GPT4Checker()
    elif args.checker_name == "nli":
        checker = NLIChecker()
    elif args.checker_name == "alignscore":
        checker = AlignScoreChecker()
    elif args.checker_name == "zephyr":
        checker = ZephyrChecker()
    else:
        raise NotImplementedError
    
    retriever = None
    if args.use_retrieval:
        if args.retriever_name == "google":
            retriever = GoogleRetriever(args.cache_dir)
        else:
            raise NotImplementedError
    
    if args.aggregator_name == "strict":
        agg_fn = strict_agg
    elif args.aggregator_name == "soft":
        agg_fn = soft_agg
    elif args.aggregator_name == "major":
        agg_fn = major_agg
    else:
        raise NotImplementedError
    
    # load data
    with open(args.input_path, "r") as fp:
        input_data = json.load(fp)

    existing_output = None
    existing_responses = []
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as fp:
            existing_output = json.load(fp)
            existing_responses = [t["response"] for t in existing_output]
    
    # check triplets
    print('Checking')
    output_data = []
    for idx, item in enumerate(tqdm(input_data)):
        assert "triplets" in item, "triplets field is required"

        response = item["response"]
        if response in existing_responses:
            print(idx, "skipped")
            continue

        triplets = item["triplets"]
        if args.use_retrieval:
            reference = retriever.retrieve(item["response"])
            item["reference"] = reference
        else:
            assert "reference" in item, \
                "reference field is required if retriever is not used."
            reference = item["reference"]
        question = item.get("question", None)
        results = [
            checker.check(t, reference, question=question)
            for t in triplets
        ]
        agg_results = agg_fn(results)
        out_item = {
            **item,
            **{
                "Y": agg_results,
                "ys": results,
            }
        }
        output_data.append(out_item)
        if idx % 20 == 0:
            with open(args.output_path, "w") as fp:
                out = output_data
                if existing_output is not None:
                    out = existing_output + output_data
                json.dump(out, fp, indent=2)

    with open(args.output_path, "w") as fp:
        out = output_data
        if existing_output is not None:
            out = existing_output + output_data
        json.dump(out, fp, indent=2)


if __name__ == "__main__":
    main()
