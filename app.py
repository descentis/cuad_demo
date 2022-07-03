from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import streamlit as st
import json
from predict import run_prediction
#import textract
from io import StringIO
import PyPDF2
import torch
import requests
import json
from random import randint
import multiprocessing
import re
from fuzzywuzzy import fuzz
import pandas as pd
import unidecode

st.set_page_config(layout="wide")

# model_list = ['akdeniz27/roberta-base-cuad',
# 			  'akdeniz27/roberta-large-cuad',
# 			  'akdeniz27/deberta-v2-xlarge-cuad']

model_list = ['akdeniz27/roberta-base-cuad']
#model_list = ['akdeniz27/deberta-v2-xlarge-cuad']
# def side_disp(option):
# 	return option.replace("akdeniz27", "CoreCLM")
#st.sidebar.header("Select CUAD Model")
#model_checkpoint = st.sidebar.radio("", model_list, format_func=side_disp)
model_checkpoint = model_list[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write("CPU:", multiprocessing.cpu_count())
if model_checkpoint == "akdeniz27/deberta-v2-xlarge-cuad": import sentencepiece
# import sentencepiece
# model_checkpoint = "akdeniz27/deberta-v2-xlarge-cuad"

#st.sidebar.write("CUAD Dataset: https://huggingface.co/datasets/cuad")
with st.sidebar:
	st.image("logo_black-orange.png", width=200)

st.markdown(f'''
    <style>
        section[data-testid="stSidebar"] .css-1siy2j7 {{width: 15rem;}}
        section[data-testid="stSidebar"] .css-e1fqkh3o3 {{width: 15rem;}}
	section[data-testid="stFileUploadDropzone"] .css-1cpxqw2 {{display: none;}}
	section[data-testid="stFileUploadDropzone"] .css-edgvbvh9 {{display: none;}}
	section[data-testid="stMarkdownContainer"] .css-stCodeBlock css-4zfol6 {{width: 60%, height:150px;}}
	section[data-testid="stMarkdownContainer"] .css-stCodeBlock css-e10mrw3y0 {{width: 60%, height:150px;}}
    </style>
''',unsafe_allow_html=True)	
	
@st.cache(allow_output_mutation=True)
def load_model():
	model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
	tokenizer = AutoTokenizer.from_pretrained(model_checkpoint , use_fast=False)
	return model, tokenizer

@st.cache(allow_output_mutation=True)
def load_questions():
	entities = ["Document Name","Parties","Agreement Date","Effective Date","Expiration Date","Renewal Term","Notice Period To Terminate Renewal","Governing Law","Most Favored Nation","Non-Compete","Exclusivity","No-Solicit Of Customers","Competitive Restriction Exception","No-Solicit Of Employees","Non-Disparagement","Termination For Convenience","Rofr/Rofo/Rofn","Change Of Control","Anti-Assignment","Revenue/Profit Sharing","Price Restrictions","Minimum Commitment","Volume Restriction","Ip Ownership Assignment","Joint Ip Ownership","License Grant","Non-Transferable License","Affiliate License-Licensor","Affiliate License-Licensee","Unlimited/All-You-Can-Eat-License","Irrevocable Or Perpetual License","Source Code Escrow","Post-Termination Services","Audit Rights","Uncapped Liability","Cap On Liability","Liquidated Damages","Warranty Duration","Insurance","Covenant Not To Sue","Third Party Beneficiary"]
	#entities.sort()
	questions = ['Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer. Details: The name of the contract',
'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract',
'Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract',
'Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer. Details: The date when the contract is effective ',
'Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer. Details: On what date will the contracts initial term expire?',
'Highlight the parts (if any) of this contract related to "Renewal Term" that should be reviewed by a lawyer. Details: What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.',
'Highlight the parts (if any) of this contract related to "Notice Period To Terminate Renewal" that should be reviewed by a lawyer. Details: What is the notice period required to terminate renewal?',
'Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer. Details: Which state/countrys law governs the interpretation of the contract?',
'Highlight the parts (if any) of this contract related to "Most Favored Nation" that should be reviewed by a lawyer. Details: Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?',
'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer. Details: Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector? ',
'Highlight the parts (if any) of this contract related to "Exclusivity" that should be reviewed by a lawyer. Details: Is there an exclusive dealing  commitment with the counterparty? This includes a commitment to procure all “requirements” from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on  collaborating or working with other parties), whether during the contract or  after the contract ends (or both).',
'Highlight the parts (if any) of this contract related to "No-Solicit Of Customers" that should be reviewed by a lawyer. Details: Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?',
'Highlight the parts (if any) of this contract related to "Competitive Restriction Exception" that should be reviewed by a lawyer. Details: This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.',
'Highlight the parts (if any) of this contract related to "No-Solicit Of Employees" that should be reviewed by a lawyer. Details: Is there a restriction on a party’s soliciting or hiring employees and/or contractors from the  counterparty, whether during the contract or after the contract ends (or both)?',
'Highlight the parts (if any) of this contract related to "Non-Disparagement" that should be reviewed by a lawyer. Details: Is there a requirement on a party not to disparage the counterparty?',
'Highlight the parts (if any) of this contract related to "Termination For Convenience" that should be reviewed by a lawyer. Details: Can a party terminate this  contract without cause (solely by giving a notice and allowing a waiting  period to expire)?',
'Highlight the parts (if any) of this contract related to "Rofr/Rofo/Rofn" that should be reviewed by a lawyer. Details: Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?',
'Highlight the parts (if any) of this contract related to "Change Of Control" that should be reviewed by a lawyer. Details: Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?',
'Highlight the parts (if any) of this contract related to "Anti-Assignment" that should be reviewed by a lawyer. Details: Is consent or notice required of a party if the contract is assigned to a third party?',
'Highlight the parts (if any) of this contract related to "Revenue/Profit Sharing" that should be reviewed by a lawyer. Details: Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?',
'Highlight the parts (if any) of this contract related to "Price Restrictions" that should be reviewed by a lawyer. Details: Is there a restriction on the  ability of a party to raise or reduce prices of technology, goods, or  services provided?',
'Highlight the parts (if any) of this contract related to "Minimum Commitment" that should be reviewed by a lawyer. Details: Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?',
'Highlight the parts (if any) of this contract related to "Volume Restriction" that should be reviewed by a lawyer. Details: Is there a fee increase or consent requirement, etc. if one party’s use of the product/services exceeds certain threshold?',
'Highlight the parts (if any) of this contract related to "Ip Ownership Assignment" that should be reviewed by a lawyer. Details: Does intellectual property created  by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?',
'Highlight the parts (if any) of this contract related to "Joint Ip Ownership" that should be reviewed by a lawyer. Details: Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?',
'Highlight the parts (if any) of this contract related to "License Grant" that should be reviewed by a lawyer. Details: Does the contract contain a license granted by one party to its counterparty?',
'Highlight the parts (if any) of this contract related to "Non-Transferable License" that should be reviewed by a lawyer. Details: Does the contract limit the ability of a party to transfer the license being granted to a third party?',
'Highlight the parts (if any) of this contract related to "Affiliate License-Licensor" that should be reviewed by a lawyer. Details: Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor? ',
'Highlight the parts (if any) of this contract related to "Affiliate License-Licensee" that should be reviewed by a lawyer. Details: Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?',
'Highlight the parts (if any) of this contract related to "Unlimited/All-You-Can-Eat-License" that should be reviewed by a lawyer. Details: Is there a clause granting one party an “enterprise,” “all you can eat” or unlimited usage license?',
'Highlight the parts (if any) of this contract related to "Irrevocable Or Perpetual License" that should be reviewed by a lawyer. Details: Does the contract contain a  license grant that is irrevocable or perpetual?',
'Highlight the parts (if any) of this contract related to "Source Code Escrow" that should be reviewed by a lawyer. Details: Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy,  insolvency, etc.)?',
'Highlight the parts (if any) of this contract related to "Post-Termination Services" that should be reviewed by a lawyer. Details: Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?',
'Highlight the parts (if any) of this contract related to "Audit Rights" that should be reviewed by a lawyer. Details: Does a party have the right to  audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?',
'Highlight the parts (if any) of this contract related to "Uncapped Liability" that should be reviewed by a lawyer. Details: Is a party’s liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.',
'Highlight the parts (if any) of this contract related to "Cap On Liability" that should be reviewed by a lawyer. Details: Does the contract include a cap on liability upon the breach of a party’s obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.',
'Highlight the parts (if any) of this contract related to "Liquidated Damages" that should be reviewed by a lawyer. Details: Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?',
'Highlight the parts (if any) of this contract related to "Warranty Duration" that should be reviewed by a lawyer. Details: What is the duration of any  warranty against defects or errors in technology, products, or services  provided under the contract?',
'Highlight the parts (if any) of this contract related to "Insurance" that should be reviewed by a lawyer. Details: Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?',
'Highlight the parts (if any) of this contract related to "Covenant Not To Sue" that should be reviewed by a lawyer. Details: Is a party restricted from contesting the validity of the counterparty’s ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?',
'Highlight the parts (if any) of this contract related to "Third Party Beneficiary" that should be reviewed by a lawyer. Details: Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?'
	]
	questions = [x for _,x in sorted(zip(entities,questions))]
	return questions

if 'key' not in st.session_state: 
	st.session_state.key = str(randint(1000, 100000000))

def clear_multi():
	st.session_state.multiselect = []
	return

model, tokenizer = load_model()
questions = load_questions()
uploaded_file = st.file_uploader("Choose a file (currently accepts pdf file format)", key=st.session_state.key)
contract = ""
if uploaded_file is not None:
	if '.txt' in uploaded_file.name:
		stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
		contract = stringio.read()
		page_num = {}

	elif '.doc' in uploaded_file.name or '.docx' in uploaded_file.name or '.pdf' in uploaded_file.name:
		#contract = textract.process(uploaded_file.name)
		try:
			pdfReader = PyPDF2.PdfFileReader(uploaded_file)
		except:
			st.write("Error in reading file, please try another contract")
		contract = ""
		page_num = {}
		for i in range(pdfReader.numPages):
			pageObj = pdfReader.getPage(i)
			contract += pageObj.extractText()
			page_num[len(contract)] = i+1
			#page_num[len(contract.replace('\n', ''))] = i+1
		#st.write(contract)
		page_num = dict(sorted(page_num.items()))
		#print(page_num)
	else:
		print("not a right format")
	
try:		
	with st.expander("Expand the contract document"):
		st.write(contract)
except:
	st.write("oops! looks like there is some issue while showing the contract!\n Don't worry the prediction will still work")
#contract = contracts[0]

st.header("Contract Review (Beta)")

def display_func(option):
	entities = ["Document Name","Parties","Agreement Date","Effective Date","Expiration Date","Renewal Term","Notice Period To Terminate Renewal","Governing Law","Most Favored Nation","Non-Compete","Exclusivity","No-Solicit Of Customers","Competitive Restriction Exception","No-Solicit Of Employees","Non-Disparagement","Termination For Convenience","Rofr/Rofo/Rofn","Change Of Control","Anti-Assignment","Revenue/Profit Sharing","Price Restrictions","Minimum Commitment","Volume Restriction","Ip Ownership Assignment","Joint Ip Ownership","License Grant","Non-Transferable License","Affiliate License-Licensor","Affiliate License-Licensee","Unlimited/All-You-Can-Eat-License","Irrevocable Or Perpetual License","Source Code Escrow","Post-Termination Services","Audit Rights","Uncapped Liability","Cap On Liability","Liquidated Damages","Warranty Duration","Insurance","Covenant Not To Sue","Third Party Beneficiary"]
	entities.sort()
	for e in entities:
		if e in option:
			return e
	

#selected_question = st.selectbox('Choose one of the 41 queries from the CUAD dataset:', questions)
selected_questions = []
selected_questions = st.multiselect('Select clause type(s) to search and review (can make multiple selections):', questions, format_func=display_func, key="multiselect")
#question_set = [questions[0], selected_question]

# for key, val in st.session_state.items():
# 	st.write(key)
col1, col2, col3 = st.columns([0.6,1,8])

with col1:
	Run_Button = st.button('Run', key=None)
with col2:
	Stop_button = st.button("Stop")
with col3:
	reset_button = st.button("Reset", on_click=clear_multi)
if reset_button and 'key' in st.session_state.keys():
	st.session_state.pop('key')
	st.experimental_rerun()

if 'boolean' not in st.session_state:
	st.session_state.boolean = False
if Stop_button:
	st.session_state.boolean = True

#st.write(st.session_state['boolean'])
#st.write(st.session_state['selected'])

# large-v2 indexed data
#indexed_data = pd.read_pickle("index.pickle")

# base mode indexing
indexed_data = pd.read_pickle("indexed_data9.pickle")
page_index = pd.read_pickle("all_pages.pickle")

indexed_falg = 0
if Run_Button == True and not len(contract)==0 and st.session_state.boolean == False and len(selected_questions) != 0:
#	for question in selected_questions:
	question_set = selected_questions
	with st.spinner('Running predictions...'):
		if st.session_state.boolean == False:
			if indexed_data.get(uploaded_file.name) is not None:
				indexed_flag = 1
				indexed_pred = indexed_data[uploaded_file.name]
				indexed_page = page_index[uploaded_file.name]
				i = 0
				for ques in question_set:
					try:
								
						ans = str(indexed_pred[ques])
						ans = unidecode.unidecode(ans)
						#ans = ans.replace("\xa0 ", " ")
# 						ans = ans.replace("\\xe2\\x80\\x99", "'")
# 						ans = ans.replace("\\xc2\\xa0", " ")
# 						ans = ans.replace("\\xe2\\x80\\x9d", " ")
# 						ans = ans.replace("\\xe2\\x80\\x9c", " ").replace("\\xcd\\xbe", " ")
						if "Document Name" in ques:
							page = 1
						else:
							page = indexed_page[ques]
# 						if 'AimmuneTherapeuticsInc' in uploaded_file.name:
# 							if 'Document Name' in ques:
# 								ans = ' '.join(ans.split(' ')[1:])
						if len(ans) != 0:
							st.write(str(i+1)+".\t"+f"Question: {ques}\n\n\tAnswer: {ans}"+" (page number: "+str(page)+")\n\n ")
						else:
							st.write(str(i + 1) + ".\t" + f"Question: {ques}\n\n\tAnswer: Couldn't find the answer\n\n")
					except:
						st.write(str(i + 1) + ".\t" + f"Question: {ques}\n\n\tAnswer: Couldn't find the answer\n\n")
					i += 1
					
			
			else:
				try:
					data = {'question':question_set, 'contract': contract}
					res = requests.post(f"https://a565-34-150-211-191.ngrok.io/predict", json=data)
					data = res.json()
					for key,val in data.items():
						st.write(key)
						st.write(data['detail'])
					predictions = data['prediction']
				except:

	# 				for key,val in data.items():
	# 					st.write(key)
					predictions = run_prediction(question_set, contract, model, tokenizer)
		else:
			st.write("Stopping the function")
			predictions = ""
	if indexed_flag == 0:
		if type(predictions) != str:
			for i, p in enumerate(predictions):
				#if i != 0: st.write(f"Question: {question_set[int(p)]}\n\nAnswer: {predictions[p]}\n\n")
	# 			answer = predictions[p].split(' ')
	# 			c = 5
	# 			k = 0
	# 			ans_data = ''
	# 			final = []
	# 			for each in answer:
	# 				if k%c == 0:
	# 					if ans_data != '':
	# 						final.append(ans_data)
	# 						ans_data = ''
	# 				else:
	# 					ans_data = ans_data+' '+each
	# 				k += 1
	# 				#print(final)

	# 			answer = '\n'.join(final)
				page = -1
				val = -1
				final_pred = ''
				if len(predictions[p]) != 0:
					pg_finder = {}
					all_ratio = []
					for j in range(0, len(contract), len(predictions[p])):
						matcher = contract[j:j+len(predictions[p])]
						ratio = fuzz.ratio(matcher, predictions[p])
						#st.write(ratio, j)
						pg_finder[ratio] = j
						all_ratio.append(ratio)

					max_ratio = max(all_ratio)
					val = pg_finder[max_ratio]
					k = val
					#st.write(contract[val:val+5])
					if contract[val-1] != '.' and contract[val] != '.':
						while True:
							if contract[k] == '.':
								break
							else:
								k -= 1
						k += 1
					e = val + len(predictions[p])
					if contract[val+len(predictions[p])] != '.' and contract[val+len(predictions[p])+1] != '.':
						while True:
							if contract[e] == '.':
								break
							e += 1

					final_pred = contract[k:e+1]
					final_pred = final_pred.replace("\t", " ").replace("\r", "").replace("\n", " ")

	# 			contract_check = re.sub(r'\n\s*\n', '', contract)
	# 			contract_check = re.sub(' +', ' ', contract_check)
	# 			cleaned_p = re.sub(' +', ' ', predictions[p])
	# 			val = contract_check.find(cleaned_p)
				#st.write(val)
				for key, v in page_num.items():
					if val <= key:
						page = v
						break

				#st.write(answer)
				if len(predictions[p]) != 0:
					st.write(str(i+1)+".\t"+f"Question: {question_set[int(p)]}\n\n\tAnswer: {predictions[p]}"+" (page number: "+str(page)+") \n\n ")
				else:
					st.write(str(i + 1) + ".\t" + f"Question: {question_set[int(p)]}\n\n\tAnswer: Couldn't find the answer\n\n")

if st.session_state.boolean == True:
	st.write("Prediction Stopped")
	st.session_state.boolean = False
	
