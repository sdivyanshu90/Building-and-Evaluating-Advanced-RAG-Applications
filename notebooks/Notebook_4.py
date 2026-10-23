# %% [markdown]
# Source notebook: Notebook_4.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 4: Auto-merging Retrieval

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
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

# %%
# Notebook cell 5
print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])

# %% [markdown]
# ## Auto-merging retrieval setup

# %%
# Notebook cell 7
from llama_index import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))

# %%
# Notebook cell 8
from llama_index.node_parser import HierarchicalNodeParser

# create the hierarchical node parser w/ default settings
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

# %%
# Notebook cell 9
nodes = node_parser.get_nodes_from_documents([document])

# %%
# Notebook cell 10
from llama_index.node_parser import get_leaf_nodes

leaf_nodes = get_leaf_nodes(nodes)
print(leaf_nodes[30].text)

# %%
# Notebook cell 11
nodes_by_id = {node.node_id: node for node in nodes}

parent_node = nodes_by_id[leaf_nodes[30].parent_node.node_id]
print(parent_node.text)

# %% [markdown]
# ### Building the index

# %%
# Notebook cell 13
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

# %%
# Notebook cell 14
from llama_index import ServiceContext

auto_merging_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    node_parser=node_parser,
)

# %%
# Notebook cell 15
from llama_index import VectorStoreIndex, StorageContext

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

automerging_index = VectorStoreIndex(
    leaf_nodes, storage_context=storage_context, service_context=auto_merging_context
)

automerging_index.storage_context.persist(persist_dir="./merging_index")

# %%
# Notebook cell 16
# This block of code is optional to check
# if an index file exist, then it will load it
# if not, it will rebuild it

import os
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import load_index_from_storage

if not os.path.exists("./merging_index"):
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            service_context=auto_merging_context
        )

    automerging_index.storage_context.persist(persist_dir="./merging_index")
else:
    automerging_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./merging_index"),
        service_context=auto_merging_context
    )

# %% [markdown]
# ### Defining the retriever and running the query engine

# %%
# Notebook cell 18
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine

automerging_retriever = automerging_index.as_retriever(
    similarity_top_k=12
)

retriever = AutoMergingRetriever(
    automerging_retriever, 
    automerging_index.storage_context, 
    verbose=True
)

rerank = SentenceTransformerRerank(top_n=6, model="BAAI/bge-reranker-base")

auto_merging_engine = RetrieverQueryEngine.from_args(
    automerging_retriever, node_postprocessors=[rerank]
)

# %%
# Notebook cell 19
auto_merging_response = auto_merging_engine.query(
    "What is the importance of networking in AI?"
)

# %%
# Notebook cell 20
from llama_index.response.notebook_utils import display_response

display_response(auto_merging_response)

# %% [markdown]
# ## Putting it all Together

# %%
# Notebook cell 22
import os

from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.node_parser import get_leaf_nodes
from llama_index import StorageContext, load_index_from_storage
from llama_index.retrievers import AutoMergingRetriever
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.query_engine import RetrieverQueryEngine


def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine

# %%
# Notebook cell 23
from llama_index.llms import OpenAI

index = build_automerging_index(
    [document],
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    save_dir="./merging_index",
)

# %%
# Notebook cell 24
query_engine = get_automerging_query_engine(index, similarity_top_k=6)

# %% [markdown]
# ## TruLens Evaluation

# %%
# Notebook cell 26
from trulens_eval import Tru

Tru().reset_database()

# %% [markdown]
# ### Two layers

# %%
# Notebook cell 28
auto_merging_index_0 = build_automerging_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index_0",
    chunk_sizes=[2048,512],
)

# %%
# Notebook cell 29
auto_merging_engine_0 = get_automerging_query_engine(
    auto_merging_index_0,
    similarity_top_k=12,
    rerank_top_n=6,
)

# %%
# Notebook cell 30
from utils import get_prebuilt_trulens_recorder

tru_recorder = get_prebuilt_trulens_recorder(
    auto_merging_engine_0,
    app_id ='app_0'
)

# %%
# Notebook cell 31
eval_questions = []
with open('generated_questions.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)

# %%
# Notebook cell 32
def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)

# %%
# Notebook cell 33
run_evals(eval_questions, tru_recorder, auto_merging_engine_0)

# %%
# Notebook cell 34
from trulens_eval import Tru

Tru().get_leaderboard(app_ids=[])

# %%
# Notebook cell 35
Tru().run_dashboard()

# %% [markdown]
# ### Three layers

# %%
# Notebook cell 37
auto_merging_index_1 = build_automerging_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index_1",
    chunk_sizes=[2048,512,128],
)

# %%
# Notebook cell 38
auto_merging_engine_1 = get_automerging_query_engine(
    auto_merging_index_1,
    similarity_top_k=12,
    rerank_top_n=6,
)

# %%
# Notebook cell 39
tru_recorder = get_prebuilt_trulens_recorder(
    auto_merging_engine_1,
    app_id ='app_1'
)

# %%
# Notebook cell 40
run_evals(eval_questions, tru_recorder, auto_merging_engine_1)

# %%
# Notebook cell 41
from trulens_eval import Tru

Tru().get_leaderboard(app_ids=[])

# %%
# Notebook cell 42
Tru().run_dashboard()
