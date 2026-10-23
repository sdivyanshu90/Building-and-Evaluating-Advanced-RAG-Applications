# %% [markdown]
# Source notebook: Notebook_1.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 1: Advanced RAG Pipeline

# %%
# Notebook cell 2
import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()

# %%
# Notebook cell 3
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

# %%
# Notebook cell 4
print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])

# %% [markdown]
# ## Basic RAG pipeline

# %%
# Notebook cell 6
from llama_index import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))

# %%
# Notebook cell 7
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)
index = VectorStoreIndex.from_documents([document],
                                        service_context=service_context)

# %%
# Notebook cell 8
query_engine = index.as_query_engine()

# %%
# Notebook cell 9
response = query_engine.query(
    "What are steps to take when finding projects to build your experience?"
)
print(str(response))

# %% [markdown]
# ## Evaluation setup using TruLens

# %%
# Notebook cell 11
eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        print(item)
        eval_questions.append(item)

# %%
# Notebook cell 12
# You can try your own question:
new_question = "What is the right AI job for me?"
eval_questions.append(new_question)

# %%
# Notebook cell 13
print(eval_questions)

# %%
# Notebook cell 14
from trulens_eval import Tru
tru = Tru()

tru.reset_database()

# %% [markdown]
# For the classroom, we've written some of the code in helper functions inside a utils.py file.  
# - You can view the utils.py file in the file directory by clicking on the "Jupyter" logo at the top of the notebook.
# - In later lessons, you'll get to work directly with the code that's currently wrapped inside these helper functions, to give you more options to customize your RAG pipeline.

# %%
# Notebook cell 16
from utils import get_prebuilt_trulens_recorder

tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                             app_id="Direct Query Engine")

# %%
# Notebook cell 17
with tru_recorder as recording:
    for question in eval_questions:
        response = query_engine.query(question)

# %%
# Notebook cell 18
records, feedback = tru.get_records_and_feedback(app_ids=[])

# %%
# Notebook cell 19
records.head()

# %%
# Notebook cell 20
# launches on http://localhost:8501/
tru.run_dashboard()

# %% [markdown]
# ## Advanced RAG pipeline

# %% [markdown]
# ### 1. Sentence Window retrieval

# %%
# Notebook cell 23
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

# %%
# Notebook cell 24
from utils import build_sentence_window_index

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)

# %%
# Notebook cell 25
from utils import get_sentence_window_query_engine

sentence_window_engine = get_sentence_window_query_engine(sentence_index)

# %%
# Notebook cell 26
window_response = sentence_window_engine.query(
    "how do I get started on a personal project in AI?"
)
print(str(window_response))

# %%
# Notebook cell 27
tru.reset_database()

tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence Window Query Engine"
)

# %%
# Notebook cell 28
for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))

# %%
# Notebook cell 29
tru.get_leaderboard(app_ids=[])

# %%
# Notebook cell 30
# launches on http://localhost:8501/
tru.run_dashboard()

# %% [markdown]
# ### 2. Auto-merging retrieval

# %%
# Notebook cell 32
from utils import build_automerging_index

automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)

# %%
# Notebook cell 33
from utils import get_automerging_query_engine

automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)

# %%
# Notebook cell 34
auto_merging_response = automerging_query_engine.query(
    "How do I build a portfolio of AI projects?"
)
print(str(auto_merging_response))

# %%
# Notebook cell 35
tru.reset_database()

tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                         app_id="Automerging Query Engine")

# %%
# Notebook cell 36
for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(response)

# %%
# Notebook cell 37
tru.get_leaderboard(app_ids=[])

# %%
# Notebook cell 38
# launches on http://localhost:8501/
tru.run_dashboard()
