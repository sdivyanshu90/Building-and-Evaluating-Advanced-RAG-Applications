# %% [markdown]
# Source notebook: Notebook_2.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 2: RAG Triad of metrics

# %%
# Notebook cell 2
import warnings
warnings.filterwarnings('ignore')

# %%
# Notebook cell 3
import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()

# %%
# Notebook cell 4
from trulens_eval import Tru

tru = Tru()
tru.reset_database()

# %%
# Notebook cell 5
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

# %%
# Notebook cell 6
from llama_index import Document

document = Document(text="\n\n".\
                    join([doc.text for doc in documents]))

# %%
# Notebook cell 7
from utils import build_sentence_window_index

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)

# %%
# Notebook cell 8
from utils import get_sentence_window_query_engine

sentence_window_engine = \
get_sentence_window_query_engine(sentence_index)

# %%
# Notebook cell 9
output = sentence_window_engine.query(
    "How do you create your AI portfolio?")
output.response

# %% [markdown]
# ## Feedback functions

# %%
# Notebook cell 11
import nest_asyncio

nest_asyncio.apply()

# %%
# Notebook cell 12
from trulens_eval import OpenAI as fOpenAI

provider = fOpenAI()

# %% [markdown]
# ### 1. Answer Relevance

# %%
# Notebook cell 14
from trulens_eval import Feedback

f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()

# %% [markdown]
# ### 2. Context Relevance

# %%
# Notebook cell 16
from trulens_eval import TruLlama

context_selection = TruLlama.select_source_nodes().node.text

# %%
# Notebook cell 17
import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)

# %%
# Notebook cell 18
import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)

# %% [markdown]
# ### 3. Groundedness

# %%
# Notebook cell 20
from trulens_eval.feedback import Groundedness

grounded = Groundedness(groundedness_provider=provider)

# %%
# Notebook cell 21
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons,
             name="Groundedness"
            )
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# %% [markdown]
# ## Evaluation of the RAG application

# %%
# Notebook cell 23
from trulens_eval import TruLlama
from trulens_eval import FeedbackMode

tru_recorder = TruLlama(
    sentence_window_engine,
    app_id="App_1",
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
)

# %%
# Notebook cell 24
eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)

# %%
# Notebook cell 25
eval_questions

# %%
# Notebook cell 26
eval_questions.append("How can I be successful in AI?")

# %%
# Notebook cell 27
eval_questions

# %%
# Notebook cell 28
for question in eval_questions:
    with tru_recorder as recording:
        sentence_window_engine.query(question)

# %%
# Notebook cell 29
records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()

# %%
# Notebook cell 30
import pandas as pd

pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]

# %%
# Notebook cell 31
tru.get_leaderboard(app_ids=[])

# %%
# Notebook cell 32
tru.run_dashboard()
