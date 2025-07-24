# 🏗️ Memorix SDK Architecture v0.3.0

## High-Level Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        APP[Your Application]
    end
    
    subgraph "Memorix SDK v0.3.0"
        API[MemoryAPI]
        CFG[ConfigManager]
        HIER[MemoryHierarchy]
        RECALL[RecallEngine]
        TRACE[TracingManager]
    end
    
    subgraph "Core Components"
        VS[VectorStore]
        EMB[Embedder]
        MS[MetadataStore]
        UPDATE[UpdateManager]
        PRIVACY[PrivacyEngine]
        AUDIT[AuditManager]
    end
    
    subgraph "Vector Store Plugins"
        FAISS[FAISS]
        QDRANT[Qdrant]
        CUSTOM[Custom]
    end
    
    subgraph "Embedder Plugins"
        OPENAI[OpenAI]
        GEMINI[Gemini]
        ST[Sentence Transformers]
    end
    
    subgraph "Metadata Store Plugins"
        SQLITE[SQLite]
        POSTGRES[PostgreSQL]
        REDIS[Redis]
        JSON[JSON File]
    end
    
    subgraph "Privacy & Audit"
        CONSENT[ConsentManager]
        ERASURE[DataEraser]
        AUDIT_LOG[AuditLogger]
    end
    
    APP --> API
    API --> CFG
    API --> CFG
    API --> HIER
    API --> RECALL
    API --> TRACE
    API --> VS
    API --> EMB
    API --> MS
    API --> UPDATE
    API --> PRIVACY
    API --> AUDIT
    
    VS --> FAISS
    VS --> QDRANT
    VS --> CUSTOM
    
    EMB --> OPENAI
    EMB --> GEMINI
    EMB --> ST
    
    MS --> SQLITE
    MS --> POSTGRES
    MS --> REDIS
    MS --> JSON
    
    PRIVACY --> CONSENT
    PRIVACY --> ERASURE
    AUDIT --> AUDIT_LOG
```

## Component Architecture

```mermaid
graph LR
    subgraph "MemoryAPI v0.3.0"
        STORE["store()"]
        RECALL["recall()"]
        UPDATE["update()"]
        DELETE["delete()"]
        STATS["get_statistics()"]
        CLEANUP["cleanup()"]
    end
    
    subgraph "VectorStore"
        VS_STORE["store()"]
        VS_SEARCH["search()"]
        VS_DELETE["delete()"]
        VS_UPDATE["update()"]
    end
    
    subgraph "Embedder"
        EMB_EMBED["embed()"]
        EMB_BATCH["embed_batch()"]
    end
    
    subgraph "MetadataStore"
        MS_STORE["store()"]
        MS_GET["get()"]
        MS_UPDATE["update()"]
        MS_DELETE["delete()"]
    end
    
    subgraph "Privacy & Audit"
        CONSENT_CHECK["validate_consent()"]
        ERASE_DATA["erase_user_data()"]
        AUDIT_LOG["log_event()"]
    end
    
    STORE --> VS_STORE
    STORE --> EMB_EMBED
    STORE --> MS_STORE
    STORE --> CONSENT_CHECK
    STORE --> AUDIT_LOG
    
    RECALL --> EMB_EMBED
    RECALL --> VS_SEARCH
    RECALL --> MS_GET
    RECALL --> AUDIT_LOG
    
    UPDATE --> EMB_EMBED
    UPDATE --> VS_UPDATE
    UPDATE --> MS_UPDATE
    UPDATE --> AUDIT_LOG
    
    DELETE --> VS_DELETE
    DELETE --> MS_DELETE
    DELETE --> AUDIT_LOG
    
    STATS --> MS_GET
    CLEANUP --> ERASE_DATA
```

## Data Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant API as MemoryAPI
    participant Emb as Embedder
    participant VS as VectorStore
    participant MS as MetadataStore
    
    App->>API: store(content, metadata)
    API->>API: validate_consent(user_id, agent_id)
    API->>Emb: embed(content)
    Emb-->>API: embedding_vector
    API->>VS: store(memory_id, embedding, content)
    API->>MS: store(memory_id, metadata)
    API->>API: log_audit_event(store_operation)
    API-->>App: memory_id
    
    App->>API: recall(query, scope, filters)
    API->>Emb: embed(query)
    Emb-->>API: query_embedding
    API->>VS: search(query_embedding, top_k)
    VS-->>API: results
    loop for each result
        API->>MS: get(memory_id)
        MS-->>API: metadata
    end
    API->>API: log_audit_event(recall_operation)
    API-->>App: enriched_results
```

## Configuration Architecture

```mermaid
graph TD
    subgraph "Configuration Sources"
        YAML[memorix_hierarchical_config.yaml]
        ENV[Environment Variables]
        CODE[Code Configuration]
    end
    
    subgraph "ConfigManager"
        LOAD["load_config()"]
        VALIDATE["validate()"]
        GET["get()"]
        SET["set()"]
    end
    
    subgraph "Component Factories"
        VS_FACTORY[VectorStore Factory]
        EMB_FACTORY[Embedder Factory]
        MS_FACTORY[MetadataStore Factory]
        PRIVACY_FACTORY[Privacy Factory]
        AUDIT_FACTORY[Audit Factory]
    end
    
    YAML --> LOAD
    ENV --> LOAD
    CODE --> LOAD
    
    LOAD --> VALIDATE
    VALIDATE --> GET
    GET --> VS_FACTORY
    GET --> EMB_FACTORY
    GET --> MS_FACTORY
    GET --> PRIVACY_FACTORY
    GET --> AUDIT_FACTORY
```

## Key Design Patterns

### 🔌 **Factory Pattern**
- Component factories create appropriate implementations based on configuration
- Easy to add new vector stores, embedders, and metadata stores

### 🎯 **Strategy Pattern**
- Different algorithms for similarity search, embedding generation, and metadata storage
- Runtime selection based on configuration

### 📋 **Template Method Pattern**
- Common interface for all component types
- Consistent API across different implementations

### ⚙️ **Configuration Pattern**
- Centralized configuration management with Pydantic validation
- Environment variable support
- YAML-based hierarchical configuration

### 🏗️ **Hierarchical Memory Pattern**
- Multi-tier memory management (STM → MTM → LTM)
- Automatic tier migration based on decay and scoring
- Pluggable decay and scoring strategies

### 🔐 **Privacy-First Pattern**
- Consent validation before any memory operation
- GDPR-compliant data erasure workflows
- Comprehensive audit trails for compliance

### 🎯 **Declarative Query Pattern**
- Structured recall queries with scoped filtering
- Tag-based, time-based, and agent-based filtering
- Semantic search with metadata enrichment

## Performance Considerations

### 🚀 **Optimizations**
- **Lazy Loading**: Components initialized on first use
- **Caching**: Embedding and metadata caching with Redis
- **Batch Operations**: Support for bulk operations
- **Connection Pooling**: Database connection management
- **FIFO Flow**: Efficient memory eviction and tier migration
- **Adaptive Decay**: Context-aware memory retention

### 📊 **Scalability**
- **Horizontal Scaling**: Stateless API design with Kubernetes
- **Vertical Scaling**: Efficient memory usage with tiered storage
- **Load Balancing**: Multiple vector store instances
- **Caching Layers**: Redis integration support
- **Multi-Tenant**: Agent and organization-level isolation
- **Edge Deployment**: Local memory with selective cloud sync

## Security Architecture

### 🔒 **Security Features**
- **API Key Management**: Secure credential handling
- **Input Validation**: Sanitization of all inputs with Pydantic
- **Access Control**: Metadata-based permissions with RBAC
- **Audit Logging**: Comprehensive operation tracking
- **Consent Management**: GDPR Article 7 compliance
- **Data Erasure**: GDPR Article 17 right to be forgotten

### 🛡️ **Best Practices**
- **Environment Variables**: No hardcoded secrets
- **HTTPS Only**: Secure API communications with TLS
- **Rate Limiting**: Protection against abuse
- **Data Encryption**: At-rest and in-transit encryption
- **Privacy by Design**: Consent validation before operations
- **Audit Trails**: Immutable logs for compliance

## Kubernetes Deployment Architecture

```mermaid
graph TB
    subgraph "External Clients"
        CLIENTS[Agents / APIs / SDK Calls]
    end
    
    subgraph "API Gateway"
        INGRESS[Ingress (Traefik / NGINX)]
    end
    
    subgraph "Memorix API Server"
        API[FastAPI / Uvicorn]
        AUTH[Authentication]
        ROUTING[Routing]
    end
    
    subgraph "Core Services"
        ORCHESTRATOR[Memory Orchestrator]
        PRIVACY[Privacy/Audit APIs]
    end
    
    subgraph "Storage Layer"
        VECTOR[Vector DB (Pinecone/Weaviate)]
        METADATA[Metadata DB (PostgreSQL)]
        AUDIT_LOG[Audit Log (Loki)]
    end
    
    CLIENTS --> INGRESS
    INGRESS --> API
    API --> AUTH
    API --> ROUTING
    API --> ORCHESTRATOR
    API --> PRIVACY
    
    ORCHESTRATOR --> VECTOR
    ORCHESTRATOR --> METADATA
    PRIVACY --> AUDIT_LOG
```

## Enterprise Deployment Architecture

```mermaid
graph TB
    subgraph "Development"
        DEV[Local Development]
        TEST[Testing Environment]
    end
    
    subgraph "Production"
        PROD[Production Environment]
        STAGING[Staging Environment]
    end
    
    subgraph "Infrastructure"
        CONTAINER[Docker Containers]
        K8S[Kubernetes]
        CLOUD[Cloud Services]
    end
    
    subgraph "Monitoring"
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
        LOKI[Loki]
    end
    
    DEV --> TEST
    TEST --> STAGING
    STAGING --> PROD
    
    PROD --> CONTAINER
    CONTAINER --> K8S
    K8S --> CLOUD
    
    PROD --> PROMETHEUS
    PROMETHEUS --> GRAFANA
    PROD --> LOKI
```

## Kubernetes Components

### 🔩 **Core Kubernetes Components**

| Component | Description |
|-----------|-------------|
| `memorix-api` | FastAPI app that receives all SDK/API traffic from agents |
| `memorix-core` | Manages memory tiers, routing, decay, vector/meta integration |
| `memorix-privacy` | Handles privacy configs, consent checks, and memory erasure APIs |
| `vector-db` | Pluggable vector store (e.g. Weaviate, Qdrant, Pinecone via service mesh) |
| `metadata-db` | SQL store (PostgreSQL, CockroachDB) for metadata & session configs |
| `audit-logger` | Collects audit logs; shipped via FluentBit → Loki, or S3 |
| `config-server` | Serves memory.yaml and privacy.yaml defaults for multi-tenant setups |

### 🧱 **Helm Chart Structure**

```
memorix-k8s/
├── charts/
│   └── memorix/
│       ├── templates/
│       │   ├── api-deployment.yaml
│       │   ├── orchestrator-deployment.yaml
│       │   ├── privacy-deployment.yaml
│       │   ├── ingress.yaml
│       │   ├── service.yaml
│       │   └── configmap.yaml
│       └── values.yaml
├── secrets/
│   ├── api-secrets.yaml
│   └── pg-creds.yaml
└── README.md
```

### ⚙️ **Key Features for Large Scale**

| Feature | How It's Handled |
|---------|------------------|
| Auto-scaling | HPA on memorix-api and memorix-core via CPU/memory |
| GPU/Embedding Workers | Dedicated embedding-worker pods for large-batch updates |
| Multi-Tenant Namespace | Label-based routing, tenant config loaded from YAML |
| Persistent Volume | For FAISS local store or fallback SQLite |
| Service Mesh | (Optional) Use Istio or Linkerd for policy & observability |
| Secrets | Mounted via sealed-secrets or external Vault |
| Consent + Audit | Real-time logs sent to Loki, memory wipe jobs run as Cron |
| Cold Storage | Offloaded to S3 or GCS via memory-archival jobs |

### 🌐 **External Connections**
- **Agents** → via SDK or API → Ingress + `/v1/memory/...` routes
- **S3 / GCS** → used by memory snapshot jobs or LTM backups
- **Kafka / NATS** → optional message bus for event replay, memory triggers
- **LangChain / RAG Frameworks** → integrate through `/recall` and `/query` endpoints

### 🔐 **Security**
- **JWT or OAuth2** agent auth
- **TLS termination** at ingress
- **RBAC** per org/agent
- **GDPR hooks** on all memory writes/reads

### 🧪 **Dev/Test Setup**

```bash
kubectl create ns memorix
helm install memorix charts/memorix/ -n memorix
kubectl port-forward svc/memorix-api 8000:80 -n memorix
```

## Integration Points

### 🔗 **External Services**
- **Vector Databases**: FAISS, Qdrant, Pinecone, Weaviate
- **Embedding APIs**: OpenAI, Google, Cohere, Hugging Face
- **Storage**: SQLite, PostgreSQL, Redis, S3
- **Monitoring**: Prometheus, Grafana, Loki, ELK Stack

### 🔌 **Framework Integrations**
- **Web Frameworks**: FastAPI, Flask, Django
- **AI Frameworks**: LangChain, LlamaIndex, Transformers
- **Cloud Platforms**: AWS, GCP, Azure
- **Orchestration**: Airflow, Prefect, Dagster 