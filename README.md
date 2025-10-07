<div align="center">
  <img src="https://github.com/user-attachments/assets/e09c8582-1791-42fc-b171-86e424535c1f" alt="Course Logo" width="400">
  <h1 align="center">Building and Evaluating Advanced RAG Applications</h1>
  <p align="center">
    A course by <strong>DeepLearning.AI</strong> in collaboration with <strong>LlamaIndex</strong> and <strong>Snowflake</strong>
  </p>
  <p align="center">
    An unofficial repository containing detailed notes and explanations of the course content.
  </p>
</div>

-----

## 📖 Introduction

Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for building knowledgeable and factual Large Language Models (LLMs). However, moving from a basic proof-of-concept to a robust, production-ready RAG system requires a sophisticated approach to both information retrieval and performance evaluation. Standard RAG pipelines, while effective, often face challenges with context precision, noise, and the "lost in the middle" problem, where vital information is buried within large, retrieved text chunks.

This course, **"Building and Evaluating Advanced RAG Applications,"** dives deep into the next generation of RAG techniques designed to overcome these limitations. You will learn how to implement advanced retrieval strategies that provide the LLM with highly relevant and coherent context, significantly improving the quality of generated responses.

Furthermore, the course emphasizes that building a great RAG system is an iterative process driven by rigorous evaluation. We will introduce the **RAG Triad** — a comprehensive framework for measuring Context Relevance, Groundedness, and Answer Relevance. These metrics provide a holistic view of your system's performance, enabling you to diagnose bottlenecks, prevent hallucinations, and systematically enhance your application.

By the end of this course, you will be equipped with the knowledge to build, evaluate, and refine advanced RAG pipelines, ensuring your applications are not only powerful but also reliable and accurate.

-----

## 📚 Course Topics Explained

This section provides an in-depth explanation of the core topics covered in the course.

1.  The Advanced RAG Pipeline
2.  The RAG Triad of Metrics
3.  Sentence-Window Retrieval
4.  Auto-Merging Retrieval

-----
<details>
  <summary><strong>The Advanced RAG Pipeline</strong></summary>

A baseline Retrieval-Augmented Generation (RAG) pipeline follows a straightforward process: it takes a user query, embeds it into a vector, and uses this vector to find the most similar text chunks from a pre-indexed knowledge base. These retrieved chunks are then concatenated and fed into a Large Language Model (LLM) along with the original query to generate a final answer. While this approach is a significant step up from relying solely on an LLM's parametric memory, it has inherent limitations that can hinder performance in real-world applications. The primary issues stem from naive chunking and retrieval strategies. Fixed-size chunking can awkwardly split concepts, while retrieving overly large chunks can introduce noise and cause the LLM to lose focus on the most critical information—a phenomenon known as the "lost in the middle" problem.

An **Advanced RAG Pipeline** seeks to optimize every stage of this process to deliver more precise, contextually-aware, and efficient results. It transforms the simple "retrieve-and-generate" model into a multi-stage, intelligent workflow. The key advancements lie in more sophisticated indexing, retrieval, and post-processing techniques. For instance, instead of fixed-size chunks, advanced pipelines might use semantic chunking, where text is divided based on conceptual shifts, ensuring that each chunk represents a coherent thought or topic. The indexing phase can also be enriched by storing detailed metadata alongside each chunk, such as document titles, section headers, or creation dates, which can be used for more targeted filtering during retrieval.

The retrieval stage is where the most significant enhancements are often made. Advanced RAG moves beyond simple vector similarity search by incorporating multiple layers of logic. One common technique is **re-ranking**. In this approach, an initial, fast retrieval process (like vector search) fetches a large set of candidate documents (e.g., the top 50). Then, a more sophisticated and computationally intensive model, like a cross-encoder, re-ranks these candidates to find the most relevant ones to pass to the LLM. Another technique involves **query transformations**, where the user's initial query is refined or expanded before retrieval. For example, a system might generate multiple sub-questions from the original query to gather a more comprehensive set of contexts or use a smaller LLM to generate a hypothetical answer to the query and use the embedding of that answer for retrieval (a technique known as HyDE). Ultimately, an advanced RAG pipeline is a modular and strategic system designed to maximize the "signal-to-noise" ratio of the context provided to the LLM, leading to more accurate, attributable, and relevant answers.
</details>

<details>
  <summary><strong>The RAG Triad of Metrics</strong></summary>

Building an effective RAG system is an empirical science that requires continuous experimentation and evaluation. Simply judging the final output as "good" or "bad" is insufficient for meaningful improvement. To systematically diagnose and enhance a RAG pipeline, we need a granular, multi-faceted evaluation framework. This is where the **RAG Triad of Metrics** comes in, providing a structured approach to measuring the performance of the core components of the system. The triad consists of three critical pillars: Context Relevance, Groundedness, and Answer Relevance. Each metric targets a different stage of the RAG process, allowing developers to pinpoint exactly where their pipeline is excelling or failing.

1.  **Context Relevance:** This is the first and arguably most important metric, as it evaluates the performance of the retrieval system itself. It answers the question: **"Is the retrieved information pertinent to the user's query?"** If the retriever fails to fetch relevant context, the subsequent generation stage is doomed from the start, a classic "garbage in, garbage out" scenario. A low context relevance score indicates a problem with the indexing strategy, the retrieval algorithm, or the query embedding model. For example, the chunks might be too broad, or the similarity search might be failing to capture the semantic nuance of the query. Evaluating context relevance involves assessing each retrieved context chunk against the original query to determine its usefulness, often using an LLM as a judge to score the alignment.

2.  **Groundedness (or Faithfulness):** This metric focuses on the relationship between the generated answer and the provided context. It answers the question: **"Is the LLM's response factually supported by the retrieved information?"** Groundedness is the primary defense against LLM hallucination within a RAG framework. An answer can be plausible and well-written, but entirely fabricated if it deviates from the source material. A low groundedness score indicates that the LLM is either ignoring the provided context or is over-relying on its parametric memory to generate information not present in the source documents. To measure groundedness, the generated answer is broken down into individual claims, and each claim is cross-referenced with the retrieved context to verify its accuracy. A high groundedness score provides confidence that the RAG system is producing trustworthy and attributable results.

3.  **Answer Relevance:** This final metric evaluates the end-to-end performance of the pipeline from the user's perspective. It answers the question: **"Does the final answer directly and comprehensively address the user's query?"** An answer can be perfectly grounded in relevant context but still be unhelpful if it is poorly phrased, evasive, or fails to address the core intent of the query. For instance, if a user asks, "What are the pros and cons of sentence-window retrieval?" and the system provides a detailed, grounded explanation of what it is but never mentions the pros and cons, it would score low on answer relevance. This metric ensures that the entire system is working in concert to deliver a high-quality, useful response that satisfies the user's need. Together, the RAG Triad provides a complete diagnostic toolkit for building robust and reliable RAG applications.
</details>

<details>
  <summary><strong>Sentence-Window Retrieval</strong></summary>

One of the most fundamental challenges in designing an RAG pipeline is determining the optimal size for text chunks. This decision involves a critical trade-off. On one hand, small, granular chunks (like individual sentences) are excellent for semantic precision. Their embeddings capture a very specific meaning, making them easier to match accurately with a user's query. However, these small chunks often lack the broader context necessary for an LLM to generate a comprehensive answer. For example, a single sentence might mention a technical term, but the surrounding sentences are needed to understand its definition and significance. On the other hand, large chunks (like entire paragraphs or pages) retain this broader context but introduce noise and dilute the semantic signal. This makes it harder for the retrieval system to pinpoint the exact piece of relevant information and increases the risk of the "lost in the middle" problem.

**Sentence-Window Retrieval** is an advanced retrieval strategy designed to resolve this trade-off by combining the best of both worlds. It enables highly precise retrieval while providing the LLM with expansive, context-rich information. The process begins during the indexing phase. Instead of creating arbitrary chunks, the document is first split into individual sentences. Each sentence is then embedded and stored as a distinct, retrievable unit. The crucial innovation is that each sentence is also stored with metadata that links it to its surrounding context. Specifically, a "window" of sentences—for example, the three sentences preceding it and the three sentences following it—is associated with that single sentence node.

The magic happens during the retrieval phase. When a query is received, the similarity search is performed against the embeddings of the *individual sentences*. This allows the system to identify the most semantically relevant sentence with high precision, effectively finding the "needle" in the haystack. However, instead of passing only this isolated sentence to the LLM, the system uses the stored metadata to retrieve the full context window associated with that sentence. If multiple relevant sentences are found, their corresponding windows are fetched and stitched together to form a larger, cohesive block of text. This augmented context is then passed to the LLM for answer generation. This method ensures that the retrieval is laser-focused on the most relevant information, while the synthesis stage benefits from the rich, surrounding context necessary for generating a nuanced and well-informed response. By decoupling the unit of retrieval (a single sentence) from the unit of synthesis (the sentence window), this technique significantly improves the quality and coherence of the generated answers.
</details>

<details>
  <summary><strong>Auto-Merging Retrieval</strong></summary>

Documents, especially long-form and structured ones like technical manuals, research papers, or legal texts, are inherently hierarchical. They are organized into chapters, sections, sub-sections, paragraphs, and finally, sentences. A standard RAG pipeline that uses naive chunking flattens this rich structure, treating every piece of text as an independent unit. This process loses the valuable relationships between different parts of the document, often leading to fragmented and incomplete context being passed to the LLM. For example, a system might retrieve several disconnected paragraphs from a single section, forcing the LLM to piece together the overarching theme without the benefit of section headers or introductory paragraphs that provide crucial context. This can result in answers that are factually correct at a micro level but lack coherence and a holistic understanding of the topic.

**Auto-Merging Retrieval**, also known as Hierarchical Retrieval, is an intelligent strategy designed to preserve and leverage the natural structure of documents. It works by creating a hierarchical index that mirrors the document's organization. During the indexing phase, the document is parsed into a tree-like structure of nodes. The smallest, most granular pieces of text (e.g., sentences or small paragraphs) form the "leaf" nodes of this tree. Larger, more comprehensive sections of the document that contain these leaf nodes become the "parent" nodes. For example, several paragraph-level leaf nodes would be grouped under a single parent node representing the sub-section they belong to. This process continues up the hierarchy, creating a multi-layered map of the document's content.

The retrieval process then operates on this hierarchical index. Initially, the system fetches the smaller, more granular leaf nodes that are most relevant to the user's query. This ensures that the retrieval is precise and targeted. The key innovation—the "auto-merging" step—comes next. The system analyzes the retrieved leaf nodes and checks if a significant number of them belong to the same parent node. For instance, a rule might be set that if more than three leaf nodes from a single parent section are retrieved, it's highly probable that the entire parent section is relevant to the query. When this threshold is met, the system automatically replaces the collection of smaller, fragmented leaf nodes with their single, larger parent node. This "merged" context, which is a complete and coherent section from the original document, is then sent to the LLM. This approach prevents the LLM from receiving disjointed snippets of information and instead provides it with a logically structured, contextually complete block of text, dramatically improving its ability to synthesize high-quality, coherent, and comprehensive answers.
</details>

-----

## 🎉 Acknowledgements

This content is based on the invaluable material provided in the **[Building and Evaluating Advanced RAG Applications](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)** short course. We extend our sincere gratitude to the teams at:

  - [**DeepLearning.AI**](https://www.deeplearning.ai/)
  - [**LlamaIndex**](https://www.llamaindex.ai/)
  - [**Snowflake**](https://www.snowflake.com/en/)

for developing and sharing this expert knowledge with the community.
