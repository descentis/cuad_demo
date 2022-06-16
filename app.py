from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import streamlit as st
import json
from predict import run_prediction

st.set_page_config(layout="wide")

model_list = ['akdeniz27/roberta-base-cuad',
			  'akdeniz27/roberta-large-cuad',
			  'akdeniz27/deberta-v2-xlarge-cuad']
st.sidebar.header("Select CUAD Model")
model_checkpoint = st.sidebar.radio("", model_list)

if model_checkpoint == "akdeniz27/deberta-v2-xlarge-cuad": import sentencepiece

st.sidebar.write("Project: https://www.atticusprojectai.org/cuad")
st.sidebar.write("Git Hub: https://github.com/TheAtticusProject/cuad")
st.sidebar.write("CUAD Dataset: https://huggingface.co/datasets/cuad")
st.sidebar.write("License: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/")

@st.cache(allow_output_mutation=True)
def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint , use_fast=False)
    return model, tokenizer

@st.cache(allow_output_mutation=True)
def load_questions():
	with open('test.json') as json_file:
		data = json.load(json_file)
		

	questions = []
	for i, q in enumerate(data['data'][0]['paragraphs'][0]['qas']):
		question = data['data'][0]['paragraphs'][0]['qas'][i]['question']
		questions.append(question)
	return questions

@st.cache(allow_output_mutation=True)
def load_contracts():
	with open('test.json') as json_file:
		data = json.load(json_file)

	contracts = []
	for i, q in enumerate(data['data']):
		contract = ' '.join(data['data'][i]['paragraphs'][0]['context'].split())
		contracts.append(contract)
	return contracts

model, tokenizer = load_model()
questions = load_questions()
contracts = load_contracts()

contract = contracts[0]

st.header("Contract Understanding Atticus Dataset (CUAD) Demo")
st.write("Based on https://github.com/marshmellow77/cuad-demo")


selected_question = st.selectbox('Choose one of the 41 queries from the CUAD dataset:', questions)
question_set = [questions[0], selected_question]

contract_type = st.radio("Select Contract", ("Sample Contract", "New Contract"))
if contract_type == "Sample Contract":
	sample_contract_num = st.slider("Select Sample Contract #")
	contract = contracts[sample_contract_num]
	with st.expander(f"Sample Contract #{sample_contract_num}"):
		st.write(contract)
else:
	contract = st.text_area("Input New Contract", "", height=256)

Run_Button = st.button("Run", key=None)
if Run_Button == True and not len(contract)==0 and not len(question_set)==0:
	predictions = run_prediction(question_set, contract, 'akdeniz27/roberta-base-cuad')
	
	for i, p in enumerate(predictions):
		if i != 0: st.write(f"Question: {question_set[int(p)]}\n\nAnswer: {predictions[p]}\n\n")
