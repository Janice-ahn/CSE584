''' This file is for running experiment for all LLMs to
##  1. running with different prompts
##  2. running the same prompt for multiple times
## which is eventually be presented as Metric: 1- F1(S_st1, {All - Labels}/ST_st2)
'''

# Imports
import json, os
import re
import sys
import math
import torch
import argparse
import transformers
from fractions import Fraction
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from itertools import combinations
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

''' Value initializations including APIs '''
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GPT4Client = OpenAI(api_key=OPENAI_API_KEY)

# input variables for asking which LLMs, which st
parser = argparse.ArgumentParser()
parser.add_argument("--llm", choices=["gpt4", "gpt4o", "llama3", "galactica", "gemini"],  default="gpt4")
# parser.add_argument("--strategy", choices=["1", "2", "multi1", "multi2"],  default="1")
parser.add_argument("--prompt", choices=["0", "1", "2"],  default="0")
parser.add_argument("--last_index",  default="-1")
args = parser.parse_args()

# setting the prompt diversity
prompt_versions = [
        "",
        "Question might not be solvable. ",
        "Make sure if the question contains error. "
    ]


'''Functions for prompting Models'''
class LLMProcessor:
    def solve_problem_with_gpt4(self, problem_text, whichPrompt):
        response = GPT4Client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
            "role": "system",
            "content": f"Solve the question that user provides. {whichPrompt}"
            },
            {
            "role": "user",
            "content": f"{problem_text}"
            }
        ],
        temperature=1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )    
        return response.choices[0].message.content 
    
    def solve_problem_with_gpt4o(self, problem_text, whichPrompt):
        response = GPT4Client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content": f"Solve the question that user provides. {whichPrompt}"
            },
            {
            "role": "user",
            "content": f"{problem_text}"
            }
        ],
        temperature=1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )    
        return response.choices[0].message.content 
    
    
    def solve_problem_with_llama3(self, problem_text, whichPrompt):
        messages = [
            {"role": "system", "content": f"Solve the question that user provides. {whichPrompt}"},
            {"role": "user", "content": f"{problem_text}" },
        ]
        
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")


        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        outputs = generation_pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response_text = outputs[0]["generated_text"][len(prompt):]
        return response_text


    def solve_problem_with_galactica(self, problem_text, whichPrompt):      
        # pip install accelerate
        tokenizer = AutoTokenizer.from_pretrained("GeorgiaTechResearchInstitute/galactica-30b-evol-instruct-70k")
        model = AutoModelForCausalLM.from_pretrained("GeorgiaTechResearchInstitute/galactica-30b-evol-instruct-70k", device_map="auto", torch_dtype=torch.bfloat16)
        
        # the evol-instruct models were fine-tuned with the same hidden prompts as the Alpaca project
        no_input_prompt_template = ("### Instruction:\n{instruction}\n\n### Response:")
        prompt = {f"Solve the question. {whichPrompt}: {problem_text}"}
                    
        formatted_prompt = no_input_prompt_template.format_map({'instruction': prompt})

        tokenized_prompt = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        out_tokens = model.generate(tokenized_prompt, max_new_tokens= 256)
        response = tokenizer.batch_decode(out_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        print(response)
        # polish the response text 
        if isinstance(response, list):  
            response = response[0]  

        marker = "\n\n### Response:"
        if marker in response:
            # print( response.split(marker, 1)[1].strip())
            return response.split(marker, 1)[1].strip()
        return "No response found."
    
    
    def solve_problem_with_gemini(self, problem_text, whichPrompt):      
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Solve the question that user provides. {whichPrompt}: {problem_text}")

        return response.text
        
    #################################################
    # TODO: Write new function for your llm         #
    #       Make sure to return response text only  #
    #################################################
    # def solve_problem_with_{llm_name_here}(self, problem_text, options):
    #     return 


#########################################################
# Functions for getting answers from the model response #



''' Functions for using other functions generated above '''
def getting_response_from_model(response_list, problem_text):
    # Solving the problem with model 
    processor = LLMProcessor()  
    method_name = f"solve_problem_with_{args.llm}"
    if hasattr(processor, method_name):
        func = getattr(processor, method_name)
        response = func(problem_text, prompt_versions[int(args.prompt)])

    response_list.append(response)   # this contains actual response from the model
    return response_list



''' main '''
def main(lastId):
    # Original Combined file
    file_path = f"./final_project_dataset.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    # File that will store all the results
    updated_file_path = f'./output/{args.llm}_P{args.prompt}.json'
    existing_content = None
    if os.path.exists(updated_file_path):
        with open(updated_file_path, 'r') as infile:
            existing_content = infile.read()
            
    # with open(updated_file_path, 'w') as outfile, open(rp_answer_order_file_path, 'w') as rp_outfile:
    with open(updated_file_path, 'w') as outfile:
        if existing_content:
            outfile.write(existing_content) # Starting from where it stopped
        if lastId == -1:
            outfile.write('[')          # All json file starts with '['   
        for problem in data:
            if problem['problem Id'] > lastId :
                problem_text = problem['Question']
                response_list = []
                response_list = getting_response_from_model(response_list, problem_text)    
                
                # Json file writing part
                if problem['problem Id'] != 0: 
                    outfile.write(',')  
                problem['response'] = response_list     
                json.dump(problem, outfile, indent=4)

        outfile.write('\n]')        # Indicating the json file has been end
    
''' Running Main function'''
lastId = int(args.last_index)
main(lastId)