# "FengZheng" Aerospace Knowledge Large Language Model

[**中文**](./README.md) | [**English**](./README_EN.md)

## 1 Introduction

As achievements in China's aerospace sector continue to emerge, public interest in aerospace knowledge has grown significantly, alongside the increasing urgency for large-scale, ongoing development of aerospace technologies. To meet the nation's and the public's needs for aerospace knowledge popularization and to assist researchers in driving continuous innovation in aerospace technology, the **Text Generation Group at the Research Center for Social Computing and Information Retrieval of Harbin Institute of Technology (HIT-SCIR-TG)** has developed the "FengZheng" aerospace knowledge large model. This model is designed primarily for young aerospace enthusiasts and professionals in the aerospace industry, leveraging artificial intelligence to integrate and promote aerospace knowledge in a more vivid and intuitive manner, enhancing the efficiency and scope of knowledge acquisition and application.  

The "FengZheng" aerospace knowledge large language model surpasses other open-source models of similar parameter sizes in terms of the proportion of aerospace knowledge within the language model and its ability to answer aerospace-related questions.  

Based on the "FengZheng" model, we have collaborated with the well-known domestic aerospace science popularization and research community, **Satellite Encyclopedia**, to launch the [Satellite Encyclopedia Aerospace Model](https://sat.huijiwiki.com/wiki/Project:大语言模型测试) on their website homepage. Since its launch at the end of June, the platform has facilitated over 7,000 aerospace-related Q&A sessions and received widespread acclaim from the user community.  

[Online Model Experience Link](http://fengzheng.hit-scir-tg.com.cn:5053/) (We are offering fast registration to the first 1,000 users with the invitation code `scir`. Other users can request an invitation code in the system, which will be sent to the corresponding email after manual review.)  

![](https://files.mdnice.com/user/76124/b1320828-28bb-4aca-92cc-e951383072ea.jpg)  

Next, we will provide a technical overview of efficient knowledge learning in domain-specific large models, model capability evaluation and results, and retrieval augmentation.  


## 2 Efficient Knowledge Learning for Aerospace Models  

Efficient continued pretraining of large language models for the aerospace domain requires high-quality data and efficient training algorithms. Pretraining data serves as the foundation for the model's capabilities, determining its performance ceiling, while training algorithms transform the knowledge within the data into the model's abilities and are key to learning efficiency. To enhance data quality and training efficiency, the construction process of the "FengZheng" model includes specific optimizations for both aspects.  

### 2.1 Acquisition of High-Quality Aerospace Documents  

The model's training is divided into two distinct phases: the knowledge injection phase and the format alignment phase.  

The **knowledge injection phase** aims to imbue the model's internal parameters with extensive factual and scientific knowledge related to aerospace. The textual data for this phase comes primarily from the internet and relevant books, including aerospace news, Wikipedia, WeChat public accounts, official websites, and textbooks and papers related to astronautics. To filter the vast amount of candidate text, we employed vocabulary-based filtering to extract documents highly relevant to the aerospace domain. The filtered text data then underwent fine-grained deduplication using the *data-juicer* tool.  

![Proportion of Data Sources in the Knowledge Injection Phase](https://files.mdnice.com/user/76124/ca45da96-40c5-49af-98ff-a3df5649ceba.png)  

The **format alignment phase** aims to enhance the model's instruction-following abilities and its capacity for multi-turn aerospace knowledge-based interactive Q&A. The interaction dialogue data for this phase comes from the *Infinite-Instruct* dataset and single-turn and multi-turn Q&A pairs generated iteratively by closed-source large language models based on high-quality aerospace domain documents. To further strengthen the model's capability to execute high-quality retrieval-augmented generation processes, we also incorporated instruction data corresponding to each node of the process.  


### 2.2 Knowledge Injection through Supervised Fine-Tuning

Continual Pre-training is a commonly used method to enhance the knowledge of domain-specific models. This approach involves training a general pre-trained model on domain-specific text data. By doing so, the model can better understand and generate text related to the domain, improving its performance on specific tasks.

The advantage of continual pre-training lies in its ability to specialize the model, increasing its accuracy and efficiency in domain-specific tasks. However, this method requires a large amount of domain-specific data and substantial computational resources. In practice, the strategy of injecting domain knowledge into a model via continual pre-training often faces inefficiency. Some domains have limited data, insufficient to support effective continual pre-training. Moreover, there is a significant format discrepancy between the continual pre-training step and subsequent supervised fine-tuning, leading to performance degradation during alignment, often referred to as the "alignment tax."

To address these challenges, we propose a strategy for knowledge injection through supervised fine-tuning. This strategy consists of two main components: the construction of multidimensional, self-supervised instruction fine-tuning samples and domain knowledge spaced training based on model-specific document perplexity ranking.

#### 2.2.1 Construction of Self-Supervised Instruction Fine-Tuning Samples

From multiple perspectives, we implement instruction fine-tuning data augmentation based on high-quality pre-trained documents. Specifically, we design a self-supervised data augmentation strategy that constructs a series of knowledge-intensive instruction fine-tuning samples from the original documents. This approach does not rely on any domain-specific processing strategies, making it applicable for training domain-specific texts in any scenario.

![Example of Self-Supervised Instruction Fine-Tuning Sample Construction](https://files.mdnice.com/user/76124/dbad3323-ba76-40a3-a458-41959861c6f1.png)

#### 2.2.2 Domain Knowledge Spaced Training Based on Model-Specific Document Perplexity Ranking

The fundamental mechanism through which large language models learn factual knowledge during pre-training involves diverse textual representations of the same factual knowledge points and the reasonable repetition of similar training samples. Based on this understanding, we introduce a domain knowledge spaced training strategy in instruction-based supervised fine-tuning for factual knowledge injection. 

Specifically, for each foundational language model (e.g., Llama3-8B), we first calculate the perplexity of each domain document in the corpus. The documents are then ranked in ascending order of perplexity, and the process of constructing self-supervised instruction fine-tuning samples, as described in the previous section, is applied sequentially. During supervised fine-tuning, the instruction data expanded from each domain document is uniformly distributed across the training data at equal intervals. Furthermore, the data in each minibatch during training maintains the original arrangement of the training data, with no shuffling applied.



### 2.3 Format Alignment Fine-Tuning

This stage aligns the model's Q&A outputs with human preferences. Training data includes single-turn and multi-turn aerospace Q&A, as well as general instruction-following data from the Infinite-Instruct dataset.

![](https://files.mdnice.com/user/76124/edea61c1-477c-49dc-a393-9e9db4ac2d91.png)

## 3 Aerospace Model Evaluation Benchmarks

In existing research on domain-specific large models, there is a lack of datasets capable of evaluating the application of large models in the aerospace domain. To address this, we constructed an evaluation benchmark tailored for aerospace. To assess the knowledge-based question-answering capability of models, the benchmark is divided into two types of tasks. The specific task definitions and data sources are as follows:  

1. **Single-point Question Answering (Single-point):** In this task, the large model is required to answer questions using a single knowledge point. The original questions are real user inquiries sourced from the Satellite Encyclopedia website, while the answers are manually annotated. The performance of the model on this task is evaluated using EM (Exact Match) and F1 scores.  

2. **Factual Long-form Question Answering (Factual-long):** In this task, the large model must integrate multiple knowledge points or factual concepts to provide a complete answer. Factual long-form questions are constructed using nouns from Satellite Encyclopedia entries, and the standard answers are derived from all attributes in these entries. The evaluation of this task is conducted by calculating the proportion of standard answers covered by the model-generated responses using string matching.



## 4 Evaluation Results

We compared the "FengZheng" model with Qwen-2-7B-Instruct, Llama3-8B-Instruct, and glm4-9B-chat. Results show that "FengZheng" outperforms in both tasks.

![Evaluation results](https://files.mdnice.com/user/76124/eefd7a95-f3a1-4677-9144-ddcafc8874ef.png)



## 5 Retrieval-Augmented Generation Enhancement

Even when trained with high-quality data, the "FengZheng" aerospace knowledge large model may produce inaccurate responses due to hallucinations or outdated information. To address these issues, we developed a Retrieval-Augmented Generation (RAG) module to enhance the capabilities of the "FengZheng" model. This module operates in three stages:  

- **Query Enhancement:** By incorporating contextual conversation history, the "FengZheng" model identifies and clarifies entities within the query, rewriting it to ensure greater clarity and precision.  
- **Query Expansion and Decomposition:** User queries often involve multiple entities. Considering the model's limitations in comprehending multi-entity queries, the query is broken down into several subqueries, each focusing on issues related to a single entity.  
- **Response Generation:** In the final stage, the model synthesizes relevant information from retrieved documents to produce coherent and informative responses.  

![Retrieval-Augmented Generation Strategy Workflow](https://files.mdnice.com/user/76124/51d17577-299e-440a-b25a-3535aba7d834.png)

### 5.1 Query Enhancement  

Multi-turn conversations often involve numerous references and entity mentions. To accurately interpret user intent and provide more satisfactory results, the large model rewrites queries. As illustrated, after receiving a query, the "FengZheng" aerospace knowledge model uses prior conversation history to resolve ambiguities and clarify references, ensuring precise interpretation of user intent.  

The "FengZheng" model focuses on aerospace knowledge, and the enhanced query undergoes three checks during this phase:  
1. Determining whether the user’s query is related to aerospace knowledge.  
2. Assessing whether the query is suitable for retrieval-augmented generation to produce a response.  
3. Verifying compliance with content requirements to avoid restricted topics such as pornography, politics, and other sensitive content.  

Only when all three conditions are met does the model proceed with retrieval-augmented generation.  

### 5.2 Query Expansion and Decomposition  

While enhanced queries address entity references, a single query may still contain multiple entities, hindering the model's ability to generate satisfactory responses and retrieve relevant documents accurately.  

To handle diverse user queries, the Query Expansion and Decomposition module adapts retrieval granularity based on query complexity. Query complexity correlates positively with the number of domain-specific knowledge entities involved. For queries involving a single domain-specific entity, the module performs reasonable expansion to enrich information about the entity's general attributes. For queries involving multiple entities, the module decomposes the query into simpler subqueries, enhancing document retrieval recall.  

Without relying on external tools, the large model itself executes query expansion and decomposition. By taking the query as input, the model rewrites it into a set of subqueries. The output is a set of `(subquery, keywords)` pairs, where the subquery field contains a list of rewritten subquery strings, and the keywords field provides the corresponding keywords for each subquery. For each subquery generated from the expanded or decomposed original query, the module performs two steps: multi-path parallel recall and two-stage re-ranking and filtering to retrieve the corresponding documents.  

### 5.3 Response Generation  

Before inputting retrieved documents into the model context, the model evaluates the relevance of the documents to their corresponding subqueries. The filtered documents are then paired with their subqueries and combined with the user’s enhanced query as input to guide response generation. If no relevant documents are retrieved for a user query, the model is prompted to provide an appropriate explanation.  


## 6 Limitations and Future Work

1. Current model interactions are focused on Chinese; future versions will include multilingual capabilities.
2. The model primarily addresses basic aerospace knowledge; future iterations will explore its application in aerospace research.



## 7 Project Details

### 7.1 Project Structure
```plaintext
├── datas                      # Benchmark datasets
│   ├── factual_long.jsonl     # Factual-long QA dataset
│   └── single_point.jsonl     # Single-point QA dataset
├── eval_factual_long.py       # Evaluate factual-long QA
├── eval_single_point.py       # Evaluate single-point QA
├── output                     # Output files
├── README_EN.md               # English README
├── README.md                  # Chinese README
└── requirements.txt           # Dependency list
```

### 7.2 Environment Setup
```bash
conda create -n fz_bench python==3.10
conda activate fz_bench
pip install -r requirements.txt
```

### 7.3 Model Evaluation

Single-point QA:
```bash
python eval_single_point.py --model_name {your_model_path}
```

Factual-long QA:
```bash
python eval_factual_long.py --model_name {your_model_path}
```



## 8 Contributors

- Supervisors: [Prof. Xiaocheng Feng](http://ir.hit.edu.cn/~xcfeng/), [Prof. Bing Qin](http://ir.hit.edu.cn/~qinb/)  
- Main Developers: Weitao Ma, Maojin Yang, Huiyi Zhang, Huixin Liu, Shuaibo Zhao  

## 9 Disclaimer

The resources related to this project are for academic research purposes only and are strictly prohibited from being used for commercial purposes. When using components involving third-party code, please strictly adhere to the corresponding open-source licenses. The content generated by the model is influenced by factors such as computational processes, randomness, and quantization precision loss, and this project cannot guarantee its accuracy. This project assumes no legal responsibility for any content output by the model and is not liable for any losses that may result from the use of related resources or output results.



## 10 Citation
If you use the data or code, please cite:
```
@misc{FengZheng2024,
    author = {},
    title = {},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{}}
}
```