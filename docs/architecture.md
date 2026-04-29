# 政务/企业文档 RAG 问答系统：架构图与流程图

> 以下图表均使用 Mermaid 语法，可直接复制到 GitHub README、飞书文档、Notion 或支持 Mermaid 的 Markdown 编辑器中渲染。

## 1. 系统总体架构图

```mermaid
flowchart LR
    U[用户/前端] -->|创建知识库/上传文档/提问| API[FastAPI API 服务]

    subgraph API_LAYER[API 层]
        API --> KBAPI[知识库管理接口]
        API --> DOCAPI[文档管理接口]
        API --> CHATAPI[问答接口 /chat]
        API --> STREAMAPI[流式问答接口 /chat/stream]
    end

    subgraph STORAGE[存储层]
        DB[(关系型数据库\nKnowledgeBase / Document)]
        FS[(本地文件存储\nuploads/)]
        ES[(Elasticsearch\nchunk_info / document_meta)]
    end

    subgraph DOC_PIPELINE[文档处理流水线]
        PARSER[PDF/Word 解析\npdfplumber / OCR / python-docx]
        SPLITTER[章/节/条结构化分块\nbreadcrumb + page_number]
        EMB[Embedding 模型\nSentence-Transformers / BGE]
        BULK[ES Bulk 写入]
    end

    subgraph RAG_PIPELINE[RAG 问答流水线]
        REWRITE[Query Rewrite\n多轮追问改写]
        BM25[BM25 关键词检索]
        VECTOR[dense_vector 向量检索]
        RRF[RRF 融合排序]
        RERANK[Reranker 精排]
        PROMPT[Prompt 组装\n上下文 + 来源]
        LLM[LLM 生成回答]
        SOURCE[引用溯源\n文档名/章节/页码/chunk]
    end

    KBAPI --> DB
    DOCAPI --> DB
    DOCAPI --> FS
    DOCAPI -->|BackgroundTasks| PARSER
    PARSER --> SPLITTER --> EMB --> BULK --> ES
    DOCAPI -->|删除文档时清理 chunk| ES

    CHATAPI --> REWRITE
    STREAMAPI --> REWRITE
    REWRITE --> BM25 --> RRF
    REWRITE --> VECTOR --> RRF
    RRF --> RERANK --> PROMPT --> LLM --> SOURCE --> U
    PROMPT --> DB
    BM25 --> ES
    VECTOR --> ES
```

## 2. 文档上传与索引构建流程图

```mermaid
flowchart TD
    A[用户上传 PDF/Word 文档] --> B{文件类型与大小校验}
    B -->|不通过| B1[返回 400 错误]
    B -->|通过| C[校验知识库是否存在]
    C -->|不存在| C1[返回 404 错误]
    C -->|存在| D[保存原始文件到 uploads/]
    D --> E[数据库写入 Document 记录\nprocess_status=pending]
    E --> F[提交 BackgroundTask]
    F --> G[后台任务读取文档信息]
    G --> H[更新状态为 processing]
    H --> I[释放数据库 Session]
    I --> J[解析文档文本\nPDF / OCR / Word]
    J --> K{文本是否为空}
    K -->|为空| K1[更新状态为 failed\n写入 error_msg]
    K -->|非空| L[按章/节/条结构化分块]
    L --> M{分块是否为空}
    M -->|为空| M1[更新状态为 failed]
    M -->|非空| N[批量生成 Embedding 向量]
    N --> O[构造 ES bulk actions]
    O --> P[批量写入 Elasticsearch chunk_info]
    P --> Q[更新 Document 状态为 completed]
```

## 3. RAG 问答流程图

```mermaid
flowchart TD
    A[用户提交 messages + knowledge_id] --> B{messages 是否为空}
    B -->|是| B1[返回 400]
    B -->|否| C{最后一条是否为 user 提问}
    C -->|否| C1[Pydantic 校验失败]
    C -->|是| D[取最后一条用户问题]
    D --> E[Query Rewrite\n结合最近对话历史改写检索 query]
    E --> F[BM25 关键词检索]
    E --> G[dense_vector 向量检索]
    F --> H[RRF 融合排序]
    G --> H
    H --> I{是否启用 Rerank}
    I -->|否| J[取融合结果 Top-K]
    I -->|是| K[Reranker 对 query-chunk 打分]
    K --> L[按 rerank_score 排序并过滤低置信结果]
    J --> M{是否有检索结果}
    L --> M
    M -->|无| M1[返回未检索到相关内容]
    M -->|有| N[批量反查文档名]
    N --> O[组装 Prompt\n参考资料 + 用户问题]
    O --> P[加入最近多轮对话历史]
    P --> Q[调用 LLM]
    Q --> R[生成答案]
    R --> S[返回 answer + sources + messages]
```

## 4. 混合检索与重排流程图

```mermaid
flowchart LR
    Q[用户问题 / 改写后问题] --> B1[BM25 检索\nmatch chunk_content]
    Q --> V1[Embedding 编码]
    V1 --> V2[ES kNN 检索\ndense_vector cosine]

    B1 --> RRF[RRF 融合\nscore += 1/(k+rank+1)]
    V2 --> RRF

    RRF --> CAND[候选 chunk 列表]
    CAND --> PAIRS[构造 query-chunk pairs]
    PAIRS --> RERANK[Reranker 相关性打分]
    RERANK --> FILTER[阈值过滤 + Top-K 截断]
    FILTER --> CTX[最终上下文]
```

## 5. 流式问答流程图

```mermaid
sequenceDiagram
    participant Client as 前端/客户端
    participant API as FastAPI /chat/stream
    participant RAG as RAG 服务
    participant ES as Elasticsearch
    participant LLM as 大模型 API

    Client->>API: POST /chat/stream
    API->>RAG: stream_chat_with_knowledge_base()
    RAG->>RAG: Query Rewrite
    RAG->>ES: BM25 + Vector Search
    ES-->>RAG: 候选 chunks
    RAG->>RAG: RRF + Rerank + Prompt
    RAG->>LLM: stream=True 调用
    loop LLM 输出增量文本
        LLM-->>RAG: delta chunk
        RAG-->>Client: SSE data: {chunk: ...}
    end
    RAG-->>Client: SSE data: {sources: [...]}
    RAG-->>Client: SSE data: [DONE]
```

## 6. 数据模型与索引关系图

```mermaid
erDiagram
    KNOWLEDGE_BASE ||--o{ DOCUMENT : contains
    KNOWLEDGE_BASE {
        int knowledge_id PK
        string title
        string category
        datetime create_dt
        datetime update_dt
    }
    DOCUMENT {
        int document_id PK
        int knowledge_id FK
        string title
        string category
        string file_path
        string file_type
        string process_status
        string error_msg
        datetime create_dt
        datetime update_dt
    }
```

```mermaid
flowchart LR
    D[Document 数据库记录] -->|document_id / knowledge_id| C[ES chunk_info]
    C --> C1[chunk_content\n用于 BM25]
    C --> C2[embedding_vector\n用于向量检索]
    C --> C3[breadcrumb\n用于章节溯源]
    C --> C4[page_number\n用于页码溯源]
```

## 7. 评估流程图

```mermaid
flowchart TD
    A[ES 中抽样 chunk] --> B[LLM 辅助生成 QA]
    B --> C[人工检查评估集]
    C --> D[读取 eval_dataset.json]
    D --> E1[实验 1：BM25]
    D --> E2[实验 2：Hybrid + RRF]
    D --> E3[实验 3：Hybrid + RRF + Rerank]
    E1 --> F[生成答案]
    E2 --> F
    E3 --> F
    F --> G[RAGAS 评估]
    G --> H[Faithfulness]
    G --> I[Answer Relevancy]
    H --> J[输出 eval_result.json]
    I --> J
```
