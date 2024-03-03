import json
import streamlit as st
import time
import boto3
import sentencepiece
import pandas as pd
from anthropic import Anthropic
CLAUDE = Anthropic()
import multiprocessing
import subprocess
import shutil
import os
import codecs
import uuid
from streamlit_chat import message
from transformers import LlamaTokenizer
import tiktoken
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.base import BaseCallbackHandler
import pandas as pd
from io import StringIO
from transformers import AutoTokenizer
import re

REDSHIFT=boto3.client('redshift-data')
with open('config.json') as f:
    config_file = json.load(f)
CLUSTER_IDENTIFIER=config_file["redshift-identifier"]
DATABASE = config_file["database-name"]
DB_USER =config_file["database-user"]
SERVERLESS=config_file['serverless']
DEBUGGING_MAX_RETRIES=config_file['debug-max-retries']

ATHENA=boto3.client('athena')
GLUE=boto3.client('glue')
S3=boto3.client('s3')
COGNITO = boto3.client('cognito-idp')
prompt_path="prompt"
MIXTRAL_ENPOINT="mixtral"


with open('pricing.json') as f:
    pricing_file = json.load(f)

from botocore.config import Config
config = Config(
    read_timeout=600,
    retries = dict(
        max_attempts = 5
    )
)
BEDROCK=boto3.client(service_name='bedrock-runtime',region_name='us-east-1',config=config)
st.set_page_config(page_icon=None, layout="wide")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
  
        self.text+=token+""        
        self.container.markdown(self.text)

if 'input_token' not in st.session_state:
    st.session_state['input_token'] = 0
if 'output_token' not in st.session_state:
    st.session_state['output_token'] = 0
if 'action_name' not in st.session_state:
    st.session_state['action_name'] = ""
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = 0

DYNAMODB_TABLE=""
DYNAMODB_USER=""
if 'chat_memory' not in st.session_state:
    if DYNAMODB_TABLE:
        chat_histories = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": DYNAMODB_USER})
        if "Item" in chat_histories:
            st.session_state['chat_memory']=chat_histories['Item']['messages']
        else:
            st.session_state['chat_memory']=[]
    else:
        st.session_state['chat_memory'] = []  
    
    
import json

@st.cache_resource
def token_counter(path):
    tokenizer = LlamaTokenizer.from_pretrained(path)
    return tokenizer

@st.cache_resource
def mistral_counter(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer

@st.cache_data
def get_tables_redshift(identifier, database,  schema,serverless,db_user=None,):
    if serverless:
        tables_ls = REDSHIFT.list_tables(
       WorkgroupName=identifier,
        Database=database,
        SchemaPattern=schema
        )
    else:
        tables_ls = REDSHIFT.list_tables(
        ClusterIdentifier=identifier,
        Database=database,
        DbUser=db_user,
        SchemaPattern=schema
        )
    return [x['name'] for x in  tables_ls['Tables']]

@st.cache_data
def get_db_redshift(identifier, database,serverless, db_user=None, ):
    if serverless:
        db_ls = REDSHIFT.list_databases(
        WorkgroupName=identifier,
        Database=database,
        )    
    else:
        db_ls = REDSHIFT.list_databases(
        ClusterIdentifier=identifier,
        Database=database,
        DbUser=db_user
        )
    return db_ls['Databases']

@st.cache_data
def get_schema_redshift(identifier, database, serverless, db_user=None,):
    if serverless:
        schema_ls = REDSHIFT.list_schemas(
        WorkgroupName=identifier,
        Database=database,

        )
    else:        
        schema_ls = REDSHIFT.list_schemas(
        ClusterIdentifier=identifier,
        Database=database,
        DbUser=db_user
        )
    return schema_ls['Schemas']

@st.cache_data
def execute_query_redshyft(sql_query, identifier, database, serverless,db_user=None):
    if serverless:
        response = REDSHIFT.execute_statement(
        WorkgroupName=identifier,
            Database=database,

            Sql=sql_query
        )
    else:        
        response = REDSHIFT.execute_statement(
            ClusterIdentifier=identifier,
            Database=database,
            DbUser=db_user,
            Sql=sql_query
        )
    return response

def single_execute_query(params,sql_query, identifier, database, question,serverless,db_user=None):
    response = execute_query_redshyft(sql_query, identifier, database,serverless, db_user)
    df=redshyft_querys(sql_query,response,question,params,identifier, 
                       database,                     
                       question,
                         db_user,)    
    return df


def execute_query_with_pagination( sql_query, identifier, database,  serverless, db_user=None,):
    results_list=[]
    if serverless:
        response_b = REDSHIFT.batch_execute_statement(
            WorkgroupName=identifier,
            Database=database,
        
            Sqls=sql_query
        ) 
    else:
        response_b = REDSHIFT.batch_execute_statement(
            ClusterIdentifier=identifier,
            Database=database,
            DbUser=db_user,
            Sqls=sql_query
        )   
    describe_b=REDSHIFT.describe_statement(
         Id=response_b['Id'],
    )       
    status=describe_b['Status']
    while status != "FINISHED":
        time.sleep(1)
        describe_b=REDSHIFT.describe_statement(
                         Id=response_b['Id'],
                    ) 
        status=describe_b['Status']
    max_attempts = 5 
    attempts = 0
    while attempts < max_attempts:
        try:
            for ids in describe_b['SubStatements']:
                result_b = REDSHIFT.get_statement_result(Id=ids['Id'])                
                results_list.append(get_redshift_table_result(result_b))
            break
        except REDSHIFT.exceptions.ResourceNotFoundException as e:
            attempts += 1
            time.sleep(2)
    return results_list

def llm_debugga(question, statement, error, params):
    model="claude" if "claude" in params["sql_model"].lower() else "mixtral" 
    with open(f"{prompt_path}/{params['engine']}/{model}-debugger.txt","r") as f:
        prompts=f.read()
    values = {
    "error":error,
    "sql":statement,
    "schema": params['schema'],
    "sample": params['sample'],
    "question":params['prompt']
    }
    prompts=prompts.format(**values)
    if "claude" == model:
        prompts=f"\n\nHuman: {prompts}\n\nAssistant:"
        
    output=query_llm(prompts,params)
    return output

def get_redshift_table_result(response):

    columns = [c['name'] for c in response['ColumnMetadata']] 
    data = []
    for r in response['Records']:
        row = []
        for col in r:
            row.append(list(col.values())[0])  
        data.append(row)
    df = pd.DataFrame(data, columns=columns)    
    return df.to_csv(index=False)

def redshyft_querys(q_s,response,prompt,params,identifier, database, question,db_user=None,): 
    max_execution=5
    debug_count=max_execution
    alert=False
    try:
        statement_result = REDSHIFT.get_statement_result(
            Id=response['Id'],
        )
    except REDSHIFT.exceptions.ResourceNotFoundException as err:  
        describe_statement=REDSHIFT.describe_statement(
             Id=response['Id'],
        )
        query_state=describe_statement['Status']  
        while query_state in ['SUBMITTED','PICKED','STARTED']:
      
            time.sleep(1)
            describe_statement=REDSHIFT.describe_statement(
                 Id=response['Id'],
            )
            query_state=describe_statement['Status']
        while (max_execution > 0 and query_state == "FAILED"):
            max_execution = max_execution - 1
            print(f"\nDEBUG TRIAL {max_execution}")
            bad_sql=describe_statement['QueryString']
            print(f"\nBAD SQL:\n{bad_sql}")                
            error=describe_statement['Error']
            print(f"\nERROR:{error}")
            print("\nDEBUGGIN...")
            cql=llm_debugga(prompt, bad_sql, error, params)            
            idx1 = cql.index('<sql>')
            idx2 = cql.index('</sql>')
            q_s=cql[idx1 + len('<sql>') + 1: idx2]
            print(f"\nDEBUGGED SQL\n {q_s}")
            ### Guardrails to prevent the LLM from altering tables
            if any(keyword in q_s for keyword in ["CREATE", "DROP", "ALTER","INSERT","UPDATE","TRUNCATE","DELETE","MERGE","REPLACE","UPSERT"]):
                alert="I AM NOT PERMITTED TO MODIFY THIS TABLE, CONTACT ADMIN."       
                alert=True
                break
            else:
                response = execute_query_redshyft(q_s, identifier, database,params['serverless'],db_user)
                describe_statement=REDSHIFT.describe_statement(
                                     Id=response['Id'],
                                )
                query_state=describe_statement['Status']
                while query_state in ['SUBMITTED','PICKED','STARTED']:
                    time.sleep(2)            
                    describe_statement=REDSHIFT.describe_statement(
                                     Id=response['Id'],
                                )
                    query_state=describe_statement['Status']
                if query_state == "FINISHED":                
                    break 
        
        if max_execution == 0 and query_state == "FAILED":
            print(f"DEBUGGING FAILED IN {str(debug_count)} ATTEMPTS")
        elif alert:
            pass
        else:           
            max_attempts = 5
            attempts = 0
            while attempts < max_attempts:
                try:
                    time.sleep(1)
                    statement_result = REDSHIFT.get_statement_result(
                        Id=response['Id']
                    )
                    break

                except REDSHIFT.exceptions.ResourceNotFoundException as e:
                    attempts += 1
                    time.sleep(5)
    if max_execution == 0 and query_state == "FAILED":
        df=f"DEBUGGING FAILED IN {str(debug_count)} ATTEMPTS. NO RESULT AVAILABLE"
    elif alert:
        df="I AM NOT PERMITTED TO MODIFY THIS TABLE, CONTACT ADMIN."     
    else:
        df=get_redshift_table_result(statement_result)
    return df, q_s

def redshift_qna(params,stream_handler=None):
    if "tables" in params:
        sql1=f"SELECT table_catalog,table_schema,table_name,column_name,ordinal_position,is_nullable,data_type FROM information_schema.columns WHERE table_schema='{params['db_schema']}'"
        sql2=[]
        for table in params['tables']:
            sql2.append(f"SELECT * from {params['db']}.{params['db_schema']}.{table} LIMIT 10")
        sqls=[sql1]+sql2        
        # st.write(sqls)
        question=params['prompt']
        results=execute_query_with_pagination(sqls, CLUSTER_IDENTIFIER, params['db'], params['serverless'],DB_USER)    
        col_names=results[0].split('\n')[0]
        observations="\n".join(sorted(results[0].split('\n')[1:])).strip()
        params['schema']=f"{col_names}\n{observations}"
        params['sample']=''
        for examples in results[1:]:
            params['sample']+=f"{examples}\n\n"
    elif "table" in params:
        sql1=f"SELECT * FROM information_schema.columns WHERE table_name='{params['table']}' AND table_schema='{params['db_schema']}'"
        sql2=f"SELECT * from {params['db']}.{params['db_schema']}.{params['table']} LIMIT 10"
        question=params['prompt']
        sqls=[sql1]+[sql2] 
        results=execute_query_with_pagination(sqls, CLUSTER_IDENTIFIER, params['db'],params['serverless'], DB_USER)    
        params['schema']=results[0]
        params['sample']=results[1]
    model="claude" if "claude" in params["sql_model"].lower() else "mixtral"
    with open(f"{prompt_path}/{params['engine']}/{model}-sql.txt","r") as f:
        prompts=f.read()
    values = {
    "schema": params['schema'],
    "sample": params['sample'],
    "question": question,
    }
    prompts=prompts.format(**values)
    if "claude" == model:
        prompts=f"\n\nHuman: {prompts}\n\nAssistant:"


    q_s=query_llm(prompts,params)    
    print(q_s)
    sql_pattern = re.compile(r'<sql>(.*?)(?:</sql>|$)', re.DOTALL)           
    sql_match = re.search(sql_pattern, q_s)
    q_s = sql_match.group(1)
    
    if any(keyword in q_s for keyword in ["CREATE", "DROP", "ALTER","INSERT","UPDATE","TRUNCATE","DELETE","MERGE","REPLACE","UPSERT"]):
        output="I AM NOT PERMITTED TO MODIFY THIS TABLE, CONTACT ADMIN."
    else:    
        output, q_s=single_execute_query(params,q_s, CLUSTER_IDENTIFIER, params['db'] ,question,params['serverless'],DB_USER)    
    input_token=CLAUDE.count_tokens(output) if "claude" in params['model_id'].lower() else mistral_counter("mistralai/Mixtral-8x7B-v0.1").encode(output)
    if ("claude" in params['model_id'].lower() and input_token>90000) or ("claude" not in params['model_id'].lower() and len(input_token)>28000):    
        csv_rows=output.split('\n')
        st.write("TOKEN TOO LARGE, CHUNKING...")
        chunk_rows=chunk_csv_rows(csv_rows, max_token_per_chunk=80000 if "claude" in params['model_id'].lower() else 20000)
        initial_summary=[]
        for chunk in chunk_rows:            
            model="claude" if "claude" in params['model_id'].lower() else "mixtral"
            with open(f"{prompt_path}/{params['engine']}/{model}-text-gen.txt","r") as f:
                prompts=f.read()
            values = {   
            "sql":q_s,
            "csv": chunk,       
            "question":question,
            }
            prompts=prompts.format(**values)
            if "claude" == model:
                prompts=f"\n\nHuman:\n{prompts}\n\nAssistant:"
            initial_summary.append(summary_llm(prompts, params, None))            

        prompts = f'''You are a helpful assistant. 
Here is a list of answers from multiple subset of a table for a given question:
<multiple_answers>
{initial_summary}
</multiple_answers>
Here is the given question:
{question}
Your job is to merge the multiple answers into a coherent single answer.'''
        if "claude" == model:
            prompts=f"\n\nHuman:\n{prompts}\n\nAssistant:"
        response=summary_llm(prompts, params, stream_handler)     
    else:        
        model="claude" if "claude" in params['model_id'].lower() else "mixtral"
        with open(f"{prompt_path}/{params['engine']}/{model}-text-gen.txt","r") as f:
            prompts=f.read()
        values = {   
        "sql":q_s,
        "csv": output,       
        "question":question,
        }
        prompts=prompts.format(**values)
        if "claude" == model:
            prompts=f"\n\nHuman:\n{prompts}\n\nAssistant:"
        response=summary_llm(prompts, params, stream_handler)
        # summary={params[0]:summary}
        print(f"Response {response}")
    return response, q_s,output


def query_llm(prompts,params):
    import json
    import boto3   
    if "claude" in params["sql_model"].lower():
        prompt={
          "prompt": prompts,
          "max_tokens_to_sample": params['sql_token'],
          "temperature": 0,
          "top_k": 50,
          # "top_p": 1,  
             "stop_sequences": []
        }
        prompt=json.dumps(prompt)
        output = BEDROCK.invoke_model(body=prompt,
                                    modelId=params['sql_model'], 
                                    accept="application/json", 
                                    contentType="application/json")
        output=output['body'].read().decode() 
        answer=json.loads(output)['completion']
        input_token=CLAUDE.count_tokens(prompts)
        output_token=CLAUDE.count_tokens(answer)
        tokens=input_token+output_token
        pricing=input_token*pricing_file[params['sql_model']]["input"]+output_token*pricing_file[params['sql_model']]["output"]
        st.session_state['output_token']+=output_token
        st.session_state['input_token']+=input_token
        st.session_state['cost']+=pricing
        return answer
    elif "mixtral" in params["sql_model"].lower():
       
        payload = {
            "inputs":prompts,
            "parameters": {"max_new_tokens": params['sql_token'],
                           # "top_p": params['top_p'], 
                           "temperature": 0.1,
                           "return_full_text": False,}
        }
        llama=boto3.client("sagemaker-runtime")
        output=llama.invoke_endpoint(Body=json.dumps(payload), EndpointName=MIXTRAL_ENPOINT,ContentType="application/json")
        answer=json.loads(output['Body'].read().decode())[0]['generated_text']
        tkn=mistral_counter("mistralai/Mistral-7B-v0.1")
        input_token=len(tkn.encode(prompts))
        output_token=len(tkn.encode(answer))       
        tokens=input_token+output_token     
        st.session_state['output_token']+=output_token
        st.session_state['input_token']+=input_token 
        return answer


def summary_llm(prompts,params, handler=None):
    import json
    
    if 'claude' in params['model_id'].lower():
        inference_modifier = { "max_tokens_to_sample": round(params['qna_token']),
          "temperature": params['temp'], 
             "stop_sequences": []        
                     }       
        
        prompt=prompts
        llm = Bedrock(model_id=params['model_id'], client=BEDROCK, model_kwargs = inference_modifier,streaming=True if handler else False,  callbacks=handler)  
        answer =llm.invoke(prompt)       
        input_token=CLAUDE.count_tokens(prompt)
        output_token=CLAUDE.count_tokens(answer)
        tokens=input_token+output_token
        pricing=input_token*pricing_file[params['sql_model']]["input"]+output_token*pricing_file[params['sql_model']]["output"]
        st.session_state['output_token']+=output_token
        st.session_state['input_token']+=input_token
        st.session_state['cost']+=pricing
        
    elif 'ai21' in params['model_id'].lower():
        
        prompt={
          "prompt":  prompts,
          "maxTokens":round(params['qna_token']),
          "temperature": params['temp'],
          # "topP":  params['top_p'], 
        }
        prompt=json.dumps(prompt)
        response = BEDROCK.invoke_model(body=prompt,
                                modelId=params['model_id'], 
                                accept="application/json", 
                                contentType="application/json")
        answer=response['body'].read().decode() 
        answer=json.loads(answer)
        input_token=len(answer['prompt']['tokens'])
        output_token=len(answer['completions'][0]['data']['tokens'])
        tokens=input_token+output_token
        pricing=input_token*pricing_file[params['sql_model']]["input"]+output_token*pricing_file[params['sql_model']]["output"]
        answer=answer['completions'][0]['data']['text']
        st.session_state['output_token']+=output_token
        st.session_state['input_token']+=input_token
        st.session_state['cost']+=pricing

    
    elif 'titan' in params['model_id'].lower():        
      
        encoding = tiktoken.get_encoding('cl100k_base')        
        prompt={
               "inputText": prompts,
               "textGenerationConfig": {
                   "maxTokenCount": params['qna_token'],     
                   "temperature":params['temp'],
                   # "topP":params['top_p'],  
                   },
            }
        prompt=json.dumps(prompt)
        response = BEDROCK.invoke_model(body=prompt,
                                modelId=params['model_id'], 
                                accept="application/json", 
                                contentType="application/json")
        answer=response['body'].read().decode()
        answer=json.loads(answer)['results'][0]['outputText']
        input_token=len(encoding.encode(prompt))
        output_token=len(encoding.encode(answer))
        pricing=input_token*pricing_file[params['sql_model']]["input"]+output_token*pricing_file[params['sql_model']]["output"]
        tokens=input_token+output_token
        st.session_state['output_token']+=output_token
        st.session_state['input_token']+=input_token     
        st.session_state['cost']+=pricing
        
    elif 'mistral' in params['model_id'].lower() :        
        import boto3
        import json
        payload = {
            "inputs":prompts,
            "parameters": {"max_new_tokens": params['qna_token'], 
                           # "top_p": params['top_p'], 
                           "temperature": params['temp'] if params['temp']>0 else 0.01,}
        }
        llama=boto3.client("sagemaker-runtime")
        output=llama.invoke_endpoint(Body=json.dumps(payload), EndpointName="jumpstart-dft-hf-llm-mistral-7b-instruct",ContentType="application/json")
        answer=json.loads(output['Body'].read().decode())[0]['generated_text']
        tkn=mistral_counter("mistralai/Mistral-7B-v0.1")
        input_token=len(tkn.encode(prompts))
        output_token=len(tkn.encode(answer))       
        tokens=input_token+output_token     
        st.session_state['output_token']+=output_token
        st.session_state['input_token']+=input_token 
    elif 'mixtral' in params['model_id'].lower():        
        import boto3
        import json
        payload = {
            "inputs":prompts,
            "parameters": {"max_new_tokens": params['qna_token'], 
                           # "top_p": params['top_p'], 
                           "temperature": params['temp'] if params['temp']>0 else 0.01,
                          "return_full_text": False,}
        }
        llama=boto3.client("sagemaker-runtime")
        output=llama.invoke_endpoint(Body=json.dumps(payload), EndpointName=MIXTRAL_ENPOINT,ContentType="application/json")
        answer=json.loads(output['Body'].read().decode())[0]['generated_text']
        tkn=mistral_counter("mistralai/Mistral-7B-v0.1")
        input_token=len(tkn.encode(prompts))
        output_token=len(tkn.encode(answer))       
        tokens=input_token+output_token     
        st.session_state['output_token']+=output_token
        st.session_state['input_token']+=input_token 

    return answer

def chunk_csv_rows(csv_rows, max_token_per_chunk=10000):
    header = csv_rows[0]  # Assuming the first row is the header
    csv_rows = csv_rows[1:]  # Remove the header from the list
    current_chunk = []
    current_token_count = 0
    chunks = []
    header_token=CLAUDE.count_tokens(header)
    for row in csv_rows:
        token = CLAUDE.count_tokens(row)  # Assuming that the row is a space-separated CSV row.
        # print(token)
        if current_token_count + token+header_token <= max_token_per_chunk:
            current_chunk.append(row)
            current_token_count += token
        else:
            if not current_chunk:
                raise ValueError("A single CSV row exceeds the specified max_token_per_chunk.")
            header_and_chunk=[header]+current_chunk
            chunks.append("\n".join([x for x in header_and_chunk]))
            current_chunk = [row]
            current_token_count = token

    if current_chunk:
        last_chunk_and_header=[header]+current_chunk
        chunks.append("\n".join([x for x in last_chunk_and_header]))

    return chunks

def llm_memory(question, params=None):
    """ This function determines the context of each new question looking at the conversation history...
        to send the appropiate question to the retriever.    
        Messages are stored in DynamoDb if a table is provided or in memory, in the absence of a provided DynamoDb table
    """
    if DYNAMODB_TABLE:
        chat_histories = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": DYNAMODB_USER})
        if "Item" in chat_histories:
            st.session_state['chat_memory']=chat_histories['Item']['messages'][-10:]
        else:
            st.session_state['chat_memory']=[]    
    
    chat_string = ""
    for entry in st.session_state['chat_memory']:
        chat_string += f"user: {entry['user']}\nassistant: {entry['assistant']}\n"
    print(chat_string)
    memory_template = f"""\n\nHuman:
Here is the history of your conversation dialogue with a user:
<history>
{chat_string}
</history>

Here is a new question from the user:
user: {question}

Your task is to determine if the question is a follow-up to the previous conversation:
- If it is, rephrase the question as an independent question while retaining the original intent.
- If it is not, respond with "no".

Remember, your role is not to answer the question!

Format your response as:
<response>
answer
</response>\n\nAssistant:"""
    if chat_string:        
        
        prompt={
          "prompt": memory_template,
          "max_tokens_to_sample": 70,
          "temperature": 0.1,
          # "top_k": 250,
          # "top_p":params['top_p'],  
             # "stop_sequences": []
        }
        prompt=json.dumps(prompt)
        output = BEDROCK.invoke_model(body=prompt, modelId='anthropic.claude-v2', accept="application/json",  contentType="application/json")        
        output=output['body'].read().decode()
        answer=json.loads(output)['completion']
        idx1 = answer.index('<response>')
        idx2 = answer.index('</response>')
        question_2=answer[idx1 + len('<response>') + 1: idx2]
        if 'no' != question_2.strip():
            question=question_2
        print(question)
        input_token=CLAUDE.count_tokens(memory_template)
        output_token=CLAUDE.count_tokens(question)
        tokens=input_token+output_token
        st.session_state['output_token']+=output_token
        st.session_state['input_token']+=input_token
        pricing=input_token*pricing_file['anthropic.claude-v2']["input"]+output_token*pricing_file['anthropic.claude-v2']["output"]
    return question

def put_db(messages):
    """Store long term chat history in DynamoDB"""    
    chat_item = {
        "UserId": DYNAMODB_USER,
        "messages": [messages]  # Assuming 'messages' is a list of dictionaries
    }
    # Check if the user already exists in the table
    existing_item = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": DYNAMODB_USER})
    # If the user exists, append new messages to the existing ones
    if "Item" in existing_item:
        existing_messages = existing_item["Item"]["messages"]
        chat_item["messages"] = existing_messages + [messages]

    response = DYNAMODB.Table(DYNAMODB_TABLE).put_item(
        Item=chat_item
    )   
    
                      
def db_chatter(params):   
    import io
    st.title('Converse with Redshift')
    for message in st.session_state.messages:
        if "role" in message.keys():
            with st.chat_message(message["role"]):  
                st.markdown(message["content"].replace("$","USD ").replace("%", " percent"))
        else:
            with st.expander(label="**Metadata**"):              
                st.dataframe(message["df"])           
                st.code(message["sql"])
                st.markdown(f"* **Elapsed Timed**: {message['time']}")                  
            
    if prompt := st.chat_input("Hello?"):
        if params["memory"]:
            prompt=llm_memory(prompt, params=None) 
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
            
        with st.chat_message("assistant"):
            time_now=time.time()
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)
            params['prompt']=prompt       
            answer, sql_state, data=redshift_qna(params,[stream_handler])
            time_diff=time.time()-time_now
            if data:
                data = io.StringIO(data)
                df=pd.read_csv(data) 
            else:
                df=pd.DataFrame()
            message_placeholder.markdown(answer.replace("$","USD ").replace("%", " percent"))
            st.session_state.messages.append({"role": "assistant", "content": answer})       
            if params["memory"]:
                chat_history={"user" :prompt,
                "assistant":answer}           
                if DYNAMODB_TABLE:
                    put_db(chat_history)
                else:
                    st.session_state['chat_memory'].append(chat_history)  
                    st.session_state['chat_memory']=st.session_state['chat_memory'][-10:]

            with st.expander(label="**Metadata**"): 
                st.dataframe(df)
                st.code(sql_state)
                st.markdown(f"* **Elapsed Timed**: {time_diff}")           
                st.session_state.messages.append({"time": time_diff,"df": df, "sql":sql_state})
        st.rerun()
          
        
def app_sidebar():
    with st.sidebar:
        # st.text_input('Input Token Used', str(st.session_state['input_token']))
        # st.text_input('Output Token Used', str(st.session_state['output_token']))
        st.metric(label="Bedrock Session Cost", value=f"${round(st.session_state['cost'],2)}")  
        st.write('### User Preference')
        temp = st.slider('Temperature', min_value=0., max_value=1., value=0.0, step=0.01)
        mem = st.checkbox('chat memory')
        query_type=st.selectbox('Query Type', ["Table Chat","DataBase Chat"])  
        models=['amazon.titan-tg1-large',
                 'amazon.titan-e1t-medium',
                 'ai21.j2-mid',
                 'ai21.j2-ultra',
                 'anthropic.claude-instant-v1',
                 'anthropic.claude-v2:1',
                 'anthropic.claude-v2']+["Mixtral","mistral"]
        model=st.selectbox('Text Model', models, index=4)  
        sql_models=['anthropic.claude-v2',
                    'anthropic.claude-instant-v1',
                    'anthropic.claude-v2:1',
                   "Mixtral"]
        sql_model=st.selectbox('SQL Model', sql_models)         
        sql_token_length = st.slider('SQL Token Length', min_value=50, max_value=2000, value=500, step=10)
        qna_length = st.slider('Text Token Length', min_value=50, max_value=2000, value=550, step=10)       
        db=get_db_redshift(CLUSTER_IDENTIFIER, DATABASE, SERVERLESS,DB_USER)
        database=st.selectbox('Select Database',options=db,index=1)   
        schm=get_schema_redshift(CLUSTER_IDENTIFIER, database,SERVERLESS, DB_USER)
        schema=st.selectbox('Select SchemaName',options=schm)#,index=6)
        tab=get_tables_redshift(CLUSTER_IDENTIFIER, database, schema,SERVERLESS,DB_USER,)
        engine="redshift" 
        if "table" in query_type.lower():
            tab=get_tables_redshift(CLUSTER_IDENTIFIER, database, schema,SERVERLESS,DB_USER,)                
            tables=st.selectbox('Select Tables',options=tab)     
            params={'sql_token':sql_token_length,'qna_token':qna_length,'table':tables,'db':database,"db_schema":schema,'temp':temp,'model_id':model, 
                    'sql_model':sql_model, "memory":mem,"engine":engine,"serverless":SERVERLESS} 
        elif "database" in query_type.lower():
            params={'sql_token':sql_token_length,'qna_token':qna_length,'db':database,'tables':tab,"db_schema":schema,'temp':temp,'model_id':model, 
                    'sql_model':sql_model, "memory":mem,"engine":engine, "serverless":SERVERLESS} 
        return params

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        try:
            cred=COGNITO.initiate_auth(
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                 "USERNAME": st.session_state["username"],
                "PASSWORD": st.session_state["password"],
                },
                ClientId='2huaice69gslptgael5patmvcn',
            )              
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
            alias=cred['ChallengeParameters']['USER_ID_FOR_SRP']
            user_info=json.loads(cred['ChallengeParameters']['userAttributes'])
            name=f"{user_info['given_name']} {user_info['family_name']}"
            st.markdown(f"Welcome back {name} 🙂")
        except COGNITO.exceptions.NotAuthorizedException as e:
            if "Incorrect username or password" in str(e):
                # st.error("Login failed")
                st.session_state["password_correct"] = False
    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True
    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User or password incorrect")
    return False

 
        
        
def main():
    if not check_password():
        st.stop()
    params=app_sidebar()
    db_chatter(params)  

if __name__ == '__main__':
    main()       
