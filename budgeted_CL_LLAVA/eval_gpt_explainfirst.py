import argparse
import json
import os

import openai
from openai import OpenAI
import time
import copy
import base64

NUM_SECONDS_TO_SLEEP = 0.5

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_eval(content: str, max_tokens: int):
    user_content = [
                        {"type":"text", "text":content['text']},
                    ]

    count = 0
    while count < 10:
        try:
            '''
            
            '''
            response = client.chat.completions.create(
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                },{
                    'role': 'user',
                    'content': user_content
                }],
                model="gpt-3.5-turbo", # TODO: Choose gpt version
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        time.sleep(NUM_SECONDS_TO_SLEEP)
        count += 1
    return "Score: 0"

def parse_score(review):
    try:
        score = review.split('\n')[-1].split(' ')[-1]
        return float(score)

    except Exception as e:
        print(e)
        print('error', review)
        return -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-r', '--result')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()
    print(f"start evaluating {args.result}")
    results_dict = json.load(open(args.result, 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
        cur_reviews = []
    if len(cur_reviews)!=0:
        print("Skip already evaluated", len(cur_reviews))
        results_dict = results_dict[len(cur_reviews):]

    review_file = open(f'{args.output}', 'a')

    # rule = {"role": "Assistant", 
    #         "prompt": "We would like to request your feedback on the performance of an AI assistant in response to the user input and ground-truth response displayed above.\
    #             The user asks the question on observing an image or mutiple images. The images replaces <image> tokens in [Input].\n\
    #             Please rate relevance, level of details, and correctness of [Assistant] compared to [Ground-Truth]. Please focus on the correctness of the reponse more than other criteria. \
    #             The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\n\
    #             Please first output a single line containing only the value indicating the score for Assistant.\n\
    #             In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."}
    
    # rule = {"role": "Assistant", 
    #         "prompt": "We would like to request your feedback on the performance of an AI assistant in response to the user input and ground-truth response displayed above.\
    #             The user asks the question on observing an image or mutiple images. The images replaces <image> tokens in [Input].\n\
    #             Please rate relevance, level of details, and correctness of [Assistant] compared to [Ground-Truth]. Please focus on the correctness of the reponse more than other criteria. \
    #             The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\n\
    #             Please output a single line containing only the value indicating the score for Assistant."}
    
    # rule = {"role": "Assistant", 
    #         "prompt": "We would like to request your feedback on the performance of an AI assistant in response to the user input and ground-truth response displayed above.\
    #             The user asks the question on observing an image or mutiple images. The images replaces <image> tokens in [Input].\n\
    #             Please rate relevance, level of details, and correctness of [Assistant] compared to [Ground-Truth]. Please focus on the correctness of the reponse more than other criteria. \
    #             The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\n\
    #             Please first provide a brief explanation of your evaluation explanation.\n\
    #             In the subsequent line, please provide a single line containing the score for Assistant, in the format of\n\
    #             Score: <an integer value on a scale 1 to 10>"}
    
    rule = {"role": "Assistant", 
            "prompt": "Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. \
The user asks the question about observing an image or multiple images. The images replace <image> tokens in [Input].\n\
Please rate the relevance and accuracy of [Assistant] compared to [Ground-Truth]. \
The assistant receives an overall score on a scale of 1 (totally wrong) to 10 (totally right), where a higher score indicates better overall performance.\n\
Please first provide a comprehensive explanation of your evaluation explanation.\n\
In the subsequent line, please provide a single line containing the score for Assistant, in the format of \n\ Score: <an integer value on a scale 1 to 10>"}
    
    handles = []
    for item in results_dict[:-1]:
        review_item = copy.deepcopy(item)
        prompt = rule['prompt']
        role = rule['role']
        content = {
            "text": (f'[Input]\n{item["input"]}\n\n'
                   f'[Ground-Truth]\n{item["gt_sentence"]}\n\n[End of Ground-Truth]\n\n'
                   f'[{role}]\n{item["sentence"]}\n\n[End of {role}]\n\n'
                   f'[System]\n{prompt}\n\n'),
                   }
        review = get_eval(content, args.max_tokens)
        scores = parse_score(review)
        review_item['content'] = review
        review_item['score'] = scores
        cur_reviews.append(review_item)
        review_file.write(json.dumps(review_item) + '\n')
        review_file.flush()
    review_file.close()
    