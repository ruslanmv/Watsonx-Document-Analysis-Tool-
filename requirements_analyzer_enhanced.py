"""
Enhanced specialist-driven requirements analysis script with evidence pre-analysis.
This is a standalone copy for testing evidence evaluation improvements.
"""

import asyncio
import os
import re
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from agent.embeddings import get_embeddings
from config import get_best_collection_name, get_chroma_db_path

# Load environment variables
load_dotenv()
API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
ENDPOINT = os.getenv("WATSONX_ENDPOINT", "https://us-south.ml.cloud.ibm.com")

# Set the variable expected by langchain_ibm
if API_KEY:
    os.environ["WATSONX_APIKEY"] = API_KEY

# Initialize reranker
try:
    from ibm_watsonx_ai.foundation_models import Rerank
    from ibm_watsonx_ai.credentials import Credentials
    
    creds = Credentials(api_key=API_KEY, url=ENDPOINT)
    reranker = Rerank(
        model_id="cross-encoder/ms-marco-minilm-l-12-v2",
        credentials=creds,
        project_id=PROJECT_ID,
    )
    print("Reranker initialized successfully")
except Exception as e:
    print(f"Reranker initialization failed: {e}")
    reranker = None

# Create dual LLMs
granite_llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct",
    url=ENDPOINT,
    project_id=PROJECT_ID,
    params={"max_new_tokens": 512, "temperature": 0.1, "top_p": 0.9},
)

llama_llm = WatsonxLLM(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=ENDPOINT,
    project_id=PROJECT_ID,
    params={"max_new_tokens": 1024, "temperature": 0.3, "top_p": 0.95},
)

# Initialize ChromaDB with auto-detection
print("Loading ChromaDB...")
try:
    embeddings = get_embeddings()
    collection_name = get_best_collection_name()
    db_path = get_chroma_db_path()
    
    # Create default database connection
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_path
    )
    document_count = db._collection.count()
    print(f"ChromaDB loaded: collection '{collection_name}' with {document_count} documents")
    
    if document_count == 0:
        print("⚠️  WARNING: No documents found in vector database!")
        print("   Please run: python scripts/ingest_docs.py --reset")
        print("   This will ingest documents from the ./documents folder")
        
except Exception as e:
    print(f"Warning: ChromaDB not available: {e}")
    print("Will simulate retrieval for testing...")
    db = None

def get_database_connection(collection_name: str = None):
    """
    Get a ChromaDB connection for a specific collection.
    If collection_name is None, uses the default connection.
    """
    if collection_name is None:
        return db  # Use default connection
    
    try:
        embeddings = get_embeddings()
        db_path = get_chroma_db_path()
        
        specific_db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=db_path
        )
        
        document_count = specific_db._collection.count()
        print(f"Connected to collection '{collection_name}' with {document_count} documents")
        
        if document_count == 0:
            print(f"⚠️  WARNING: Collection '{collection_name}' is empty!")
            
        return specific_db
        
    except Exception as e:
        print(f"Warning: Could not connect to collection '{collection_name}': {e}")
        print("Falling back to default collection...")
        return db  # Fallback to default

# STEP 1: EVIDENCE QUERY GENERATION PROMPTS

security_evidence_queries_granite = PromptTemplate.from_template(
    """System:
You are a cybersecurity specialist with deep expertise in enterprise security architecture, compliance frameworks, and threat modeling. Your role is to identify specific evidence about security implementations, vulnerabilities, and compliance requirements.

As a security specialist, you focus on:
- Technical security controls and their implementation details
- Compliance requirements (SOX, PCI-DSS, GDPR, SOC 2, ISO 27001)
- Threat vectors and vulnerability assessments
- Security architecture patterns and best practices
- Risk assessment and mitigation strategies

Question:
Generate 2-3 short search phrases for this security requirement:

Requirement: {requirement}
Priority: {priority}

Generate search phrases that would appear in documentation about:
1. Technical implementation and standards
2. Security compliance requirements 
3. Potential risks or vulnerabilities

Format as short phrases (3-8 words each), one per line, like documentation headings or key concepts.

Answer:"""
)

performance_evidence_queries_granite = PromptTemplate.from_template(
    """System:
You are a performance engineering specialist with expertise in scalability, optimization, and system capacity planning. Your role is to identify specific evidence about system performance, bottlenecks, and optimization strategies.

As a performance specialist, you focus on:
- System benchmarks, load testing results, and capacity metrics
- Performance optimization techniques and architectural patterns
- Resource utilization (CPU, memory, network, storage)
- Scalability patterns (horizontal/vertical scaling, caching, CDN)
- SLA requirements and performance monitoring

Question:
Generate 2-3 short search phrases for this performance requirement:

Requirement: {requirement}
Priority: {priority}

Generate search phrases that would appear in documentation about:
1. System capacity and benchmarks
2. Performance optimization techniques
3. Resource requirements and scaling

Format as short phrases (3-8 words each), one per line, like documentation headings or key concepts.

Answer:"""
)

# STEP 1.5: NEW EVIDENCE PRE-ANALYSIS PROMPTS

security_evidence_preanalysis_granite = PromptTemplate.from_template(
    """System:
You are a cybersecurity specialist reviewing evidence found for a security requirement. Your job is to pre-analyze the evidence quality before final analysis.

Question:
Review the evidence found for this security requirement and provide a detailed assessment:

Requirement: {requirement}
Priority: {priority}

Evidence Found:
{evidence_content}

Evidence Sources:
{evidence_sources}

Search Queries Used:
{search_queries}

Provide a structured evidence assessment:

EVIDENCE QUALITY: [EXCELLENT/GOOD/FAIR/POOR/INSUFFICIENT]
SUPPORTING EVIDENCE: [List specific evidence that supports the requirement]
CONTRADICTING EVIDENCE: [List any evidence that contradicts or creates concerns]
EVIDENCE GAPS: [What key information is missing?]
CONFIDENCE LEVEL: [HIGH/MEDIUM/LOW] - How confident are you in this evidence?

Evidence Analysis:
- Relevance of found evidence to the requirement
- Specific technical details or standards mentioned
- Any security risks or compliance issues identified
- Quality and credibility of the evidence sources
- Recommendations for evidence interpretation

Answer:"""
)

functional_evidence_preanalysis_llama = PromptTemplate.from_template(
    """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a senior business analyst reviewing evidence found for a functional requirement. Your job is to pre-analyze the evidence quality before final analysis.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Review the evidence found for this functional requirement and provide a detailed assessment:

Requirement: {requirement}
Priority: {priority}

Evidence Found:
{evidence_content}

Evidence Sources:
{evidence_sources}

Search Queries Used:
{search_queries}

Provide a structured evidence assessment:

EVIDENCE QUALITY: [EXCELLENT/GOOD/FAIR/POOR/INSUFFICIENT]
SUPPORTING EVIDENCE: [List specific evidence that supports the requirement]
CONTRADICTING EVIDENCE: [List any evidence that contradicts or creates concerns]
EVIDENCE GAPS: [What key information is missing?]
CONFIDENCE LEVEL: [HIGH/MEDIUM/LOW] - How confident are you in this evidence?

Evidence Analysis:
- Business value and user impact mentioned in evidence
- Implementation complexity indicators
- User workflow or process implications
- Quality and relevance of the evidence sources
- Recommendations for evidence interpretation

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
)

# STEP 2: ENHANCED FINAL ANALYSIS PROMPTS (with evidence pre-analysis)

security_analysis_enhanced_granite = PromptTemplate.from_template(
    """System:
You are an intelligent AI security engineering assistant. Analyze the security requirement using both the gathered evidence and the specialist's evidence pre-analysis.

Question:
Analyze this security requirement using the gathered evidence and specialist assessment:

Requirement: {requirement}
Priority: {priority}

Evidence Pre-Analysis by Security Specialist:
{evidence_preanalysis}

Raw Evidence:
{retrieved_info}

Evidence Sources:
{sources}

Provide analysis with scores and evidence assessment:

FEASIBILITY: [0.0-1.0] - How technically feasible is this requirement?
RISK: [0.0-1.0] - How risky is implementing this requirement?
SUPPORT: [STRONG_YES/WEAK_YES/UNSURE/WEAK_NO/STRONG_NO] - How well does the evidence support this requirement?

Technical Analysis:
- Final implementation approach considering specialist evidence assessment
- Security standards and compliance based on evidence quality
- Risk assessment incorporating evidence gaps and contradictions
- Recommendations accounting for evidence limitations
- Specific next steps based on evidence confidence level

Answer:"""
)

# CONFIGURATION
EVIDENCE_QUERY_GENERATORS = {
    "security": {"prompt": security_evidence_queries_granite, "model": "granite"},
    "performance": {"prompt": performance_evidence_queries_granite, "model": "granite"},
    "functional": {"prompt": security_evidence_queries_granite, "model": "granite"},  # Reusing for now
}

EVIDENCE_PREANALYSIS_PROMPTS = {
    "security": {"prompt": security_evidence_preanalysis_granite, "model": "granite"},
    "performance": {"prompt": security_evidence_preanalysis_granite, "model": "granite"},  # Reusing for now
    "functional": {"prompt": functional_evidence_preanalysis_llama, "model": "llama"},
}

ANALYSIS_PROMPTS = {
    "security": {"prompt": security_analysis_enhanced_granite, "model": "granite"},
    "performance": {"prompt": security_analysis_enhanced_granite, "model": "granite"},  # Reusing for now
    "functional": {"prompt": security_analysis_enhanced_granite, "model": "granite"},  # Reusing for now
}

# UTILITY FUNCTIONS
def rerank_passages(query: str, passages: List[str], max_results: int = 5) -> List[Tuple[int, float, str]]:
    """Rerank passages using IBM watsonx reranker with top-k selection (no hard threshold)"""
    print(f"    🔄 RERANKING ANALYSIS")
    print(f"    ├─ Query: '{query[:60]}...'" if len(query) > 60 else f"    ├─ Query: '{query}'")
    print(f"    ├─ Input passages: {len(passages)}")
    print(f"    ├─ Strategy: TOP-{max_results} (no hard threshold)")
    print(f"    └─ Note: Scores are logits (-15 to +5 range, higher = better)")
    
    if not reranker or len(passages) <= 1:
        print(f"    ⚠️  No reranker available or insufficient passages, using default scores")
        return [(i, 0.5, passage) for i, passage in enumerate(passages)]
    
    try:
        # More generous truncation to capture more context (500 chars instead of 300)
        truncated_passages = [passage[:500] + "..." if len(passage) > 500 else passage for passage in passages]
        
        print(f"    📊 Sending to watsonx reranker (truncated to 500 chars)...")
        response = reranker.generate(query=query, inputs=truncated_passages)
        
        print(f"    📈 DETAILED RERANKING RESULTS:")
        
        # Collect all results with scores
        all_results = [(r["index"], r["score"], passages[r["index"]]) for r in response["results"]]
        
        # Sort by score (highest first) and show all results
        sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
        
        # Show all results for transparency
        for i, (idx, score, passage) in enumerate(sorted_results):
            passage_preview = passage[:100].replace('\n', ' ') + "..."
            rank_indicator = f"#{i+1}" if i < max_results else f"#{i+1} (not selected)"
            status = "✅ SELECTED" if i < max_results else "⚫ NOT SELECTED"
            print(f"      {status} - {rank_indicator}: Index {idx}, Score {score:.3f}")
            print(f"        └─ Preview: {passage_preview}")
        
        # Take top-k results
        final_results = sorted_results[:max_results]
        
        print(f"    🎯 RERANKING SUMMARY:")
        print(f"    ├─ Total passages: {len(passages)}")
        print(f"    ├─ Score range: {sorted_results[-1][1]:.3f} (worst) to {sorted_results[0][1]:.3f} (best)")
        print(f"    └─ Selected top-{len(final_results)} passages")
        
        if final_results:
            print(f"    📋 FINAL SELECTED EVIDENCE:")
            for i, (idx, score, _) in enumerate(final_results):
                print(f"      #{i+1}: Original Index {idx}, Score {score:.3f}")
        
        return final_results
        
    except Exception as e:
        print(f"    ❌ Reranking failed: {e}")
        print(f"    🔄 Falling back to default scores for all passages")
        return [(i, 0.5, passage) for i, passage in enumerate(passages)]

def get_category_simulation(requirement: str, category: str) -> str:
    """Simulate document retrieval with category-specific content"""
    simulations = {
        "security": {
            "encryption": "Security Standard: AES-256 encryption required for data at rest. TLS 1.3 for data in transit. Key management via HSM. Compliance: SOC 2 Type II, ISO 27001 certified infrastructure.",
            "authentication": "Authentication Policy: Multi-factor authentication mandatory for admin. OAuth 2.0/OIDC support. Password policy: 12+ characters. Session timeout: 30 minutes.",
            "access": "Access Control: Role-based access control (RBAC). Principle of least privilege. Regular access reviews quarterly. Audit logging for all access events."
        },
        "performance": {
            "concurrent": "Performance Benchmarks: Current system handles 200 concurrent users at 95th percentile 1.2s response time. Auto-scaling enabled at 80% CPU.",
            "load": "Load Testing Results: Peak load 1000 RPS sustainable. CDN reduces load time by 40%. Database queries optimized, 99% under 100ms.",
            "response": "SLA Requirements: 99.9% uptime target. Page load under 3s. API response under 500ms. Error rate below 0.1%."
        },
        "functional": {
            "user": "User Management: User registration with email verification. Profile management with data validation. Password reset workflows.",
            "workflow": "Business Workflows: Approval processes with configurable rules. Notification system via email/SMS. Audit trail for all actions.",
            "reporting": "Reporting Features: Standard reports with filtering. Custom report builder. Export to PDF/Excel. Scheduled report delivery."
        }
    }
    
    requirement_lower = requirement.lower()
    category_sims = simulations.get(category, {})
    
    for keyword, doc in category_sims.items():
        if keyword in requirement_lower:
            return doc
    
    return f"General {category} documentation: Industry best practices and standard implementation guidelines."

def extract_scores_with_support(analysis_text: str) -> Dict[str, Any]:
    """Extract numerical scores and support grading from analysis text"""
    scores = {}
    
    feasibility_match = re.search(r'FEASIBILITY:\s*(\d+\.?\d*)', analysis_text, re.IGNORECASE)
    if feasibility_match:
        scores['feasibility'] = float(feasibility_match.group(1))
    
    risk_match = re.search(r'RISK:\s*(\d+\.?\d*)', analysis_text, re.IGNORECASE)
    if risk_match:
        scores['risk'] = float(risk_match.group(1))
    
    support_match = re.search(r'SUPPORT:\s*\[?(STRONG_YES|WEAK_YES|UNSURE|WEAK_NO|STRONG_NO)\]?', analysis_text, re.IGNORECASE)
    if support_match:
        scores['support_grade'] = support_match.group(1).upper()
    
    return scores

# ENHANCED MAIN ANALYSIS FUNCTIONS

async def get_evidence_with_specialist_queries(requirement_dict: Dict[str, str], category: str, collection_name: str = None) -> Dict[str, Any]:
    """Generate specialist evidence queries, then search, deduplicate, and rerank"""
    requirement_text = requirement_dict['text']
    priority = requirement_dict['priority']
    
    print(f"  🔍 EVIDENCE GATHERING for {category.upper()}")
    print(f"  ═══════════════════════════════════════")
    print(f"  📋 Requirement: {requirement_text}")
    print(f"  🎯 Priority: {priority}")
    
    # Get the appropriate database connection
    current_db = get_database_connection(collection_name)
    if current_db:
        doc_count = current_db._collection.count()
        print(f"  📚 Database: {doc_count} documents in collection '{collection_name or 'default'}'")
    else:
        print(f"  📚 Database: Using simulation mode")
    
    # STEP 1: Generate evidence queries using specialist
    query_config = EVIDENCE_QUERY_GENERATORS.get(category)
    if not query_config:
        print(f"  ⚠️  No specialist configured for category: {category}")
        sim_content = get_category_simulation(requirement_text, category)
        return {
            "passages": [sim_content],
            "sources": ["Simulated Documentation"],
            "similarity_scores": [0.8],
            "rerank_scores": [0.8],
            "combined_text": sim_content,
            "evidence_queries": ["General documentation search"]
        }
    
    model = granite_llm if query_config["model"] == "granite" else llama_llm
    print(f"  🤖 Using {query_config['model'].upper()} model for {category} specialist")
    
    try:
        print(f"  💭 Invoking specialist to generate evidence queries...")
        evidence_queries_response = (query_config["prompt"] | model).invoke({
            "requirement": requirement_text,
            "priority": priority
        })
        
        # Parse search phrases - make them more search-like
        raw_queries = [q.strip("- *").strip() for q in evidence_queries_response.split("\n") 
                      if q.strip() and len(q.strip()) > 5 and len(q.strip()) < 100]
        
        # Clean up queries to be more search-friendly
        evidence_queries = []
        for query in raw_queries[:3]:  # Limit to 3
            # Remove numbering and make more search-like
            clean_query = re.sub(r'^\d+\.\s*', '', query)
            clean_query = re.sub(r'\s+', ' ', clean_query).strip()
            if len(clean_query) > 5:
                evidence_queries.append(clean_query)
        
        print(f"  🎯 Generated Evidence Queries:")
        for i, query in enumerate(evidence_queries):
            print(f"     {i+1}. '{query}'")
        
        # STEP 2: Search for each evidence query and collect ALL results
        all_chunks = {}  # Use dict to automatically deduplicate by content hash
        query_results = []  # Track results per query for logging
        
        for query_idx, query in enumerate(evidence_queries):
            print(f"\n  📊 VECTOR SEARCH {query_idx+1}/{len(evidence_queries)}")
            print(f"  ┌─ Query: '{query}'")
            
            if not current_db:
                print(f"  └─ 🔧 Using simulation fallback")
                sim_content = get_category_simulation(requirement_text, category)
                content_hash = hash(sim_content[:100])  # Simple dedup
                all_chunks[content_hash] = {
                    'content': sim_content,
                    'source': 'Simulated Documentation',
                    'similarity_score': 0.8,
                    'query': query
                }
                query_results.append({
                    'query': query,
                    'results': 1,
                    'similarity_range': '0.800'
                })
                continue
            
            try:
                # Vector search for this specific evidence query
                print(f"  ├─ 🔍 Running vector search (k=8)...")
                docs_with_scores = current_db.similarity_search_with_score(query, k=8)
                
                print(f"  ├─ 📊 Vector search results ({len(docs_with_scores)} found):")
                print(f"  │    💡 NOTE: Scores are EUCLIDEAN DISTANCE (lower = more relevant)")
                query_chunks = []
                distances = []
                
                for i, (doc, distance_score) in enumerate(docs_with_scores):
                    passage = doc.page_content[:800]  # Keep reasonable length
                    source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
                    distances.append(distance_score)
                    
                    # Create content hash for deduplication (first 200 chars should be unique enough)
                    content_hash = hash(passage[:200])
                    
                    # Store in global chunks dict (automatically deduplicates)
                    if content_hash not in all_chunks:
                        all_chunks[content_hash] = {
                            'content': passage,
                            'source': source,
                            'distance_score': distance_score,
                            'query': query,
                            'original_index': i
                        }
                        status = "🆕 NEW"
                    else:
                        # Keep the LOWER distance score (better)
                        if distance_score < all_chunks[content_hash]['distance_score']:
                            all_chunks[content_hash]['distance_score'] = distance_score
                            all_chunks[content_hash]['query'] = query
                        status = "🔄 DUPLICATE"
                    
                    preview = passage[:80].replace('\n', ' ') + "..."
                    print(f"  │    #{i+1}: Distance {distance_score:.3f} {status} - {source}")
                    print(f"  │        Preview: {preview}")
                    
                    query_chunks.append((passage, source, distance_score))
                
                # Log query summary
                if distances:
                    min_dist, max_dist = min(distances), max(distances)
                    print(f"  └─ 📈 Distance range: {min_dist:.3f} (best) - {max_dist:.3f} (worst)")
                    query_results.append({
                        'query': query,
                        'results': len(docs_with_scores),
                        'distance_range': f"{min_dist:.3f} - {max_dist:.3f}",
                        'new_chunks': len([h for h in all_chunks.keys() if all_chunks[h]['query'] == query])
                    })
                else:
                    print(f"  └─ ❌ No results found")
                    query_results.append({
                        'query': query,
                        'results': 0,
                        'distance_range': 'N/A'
                    })
                
            except Exception as e:
                print(f"  └─ 💥 Search failed: {e}")
                query_results.append({
                    'query': query,
                    'results': 0,
                    'distance_range': 'ERROR'
                })
        
        # STEP 3: Deduplication Summary
        print(f"\n  🔄 DEDUPLICATION SUMMARY:")
        print(f"  ├─ Total queries executed: {len(evidence_queries)}")
        for i, result in enumerate(query_results):
            print(f"  ├─ Query {i+1}: {result['results']} results, distance {result['distance_range']}")
        print(f"  └─ Unique chunks after deduplication: {len(all_chunks)}")
        
        if not all_chunks:
            print(f"  ⚠️  No evidence found, falling back to simulation")
            sim_content = get_category_simulation(requirement_text, category)
            return {
                "passages": [sim_content],
                "sources": ["Simulated Documentation"],
                "similarity_scores": [0.8],
                "rerank_scores": [0.8],
                "combined_text": sim_content,
                "evidence_queries": evidence_queries
            }
        
        # STEP 4: Prepare for reranking
        unique_passages = [chunk['content'] for chunk in all_chunks.values()]
        unique_sources = [chunk['source'] for chunk in all_chunks.values()]
        unique_similarities = [chunk['distance_score'] for chunk in all_chunks.values()]
        
        print(f"\n  🔄 RERANKING PHASE:")
        print(f"  ├─ Input: {len(unique_passages)} unique passages")
        print(f"  ├─ Best similarity: {min(unique_similarities):.3f}")
        print(f"  └─ Worst similarity: {max(unique_similarities):.3f}")
        
        # STEP 5: Rerank the deduplicated chunks
        reranked = rerank_passages(requirement_text, unique_passages, max_results=5)
        
        # STEP 6: Combine final evidence
        final_passages = []
        final_sources = []
        final_similarities = []
        final_rerank_scores = []
        
        for idx, rerank_score, passage in reranked:
            final_passages.append(passage)
            final_sources.append(unique_sources[idx])
            final_similarities.append(unique_similarities[idx])
            final_rerank_scores.append(rerank_score)
        
        print(f"\n  📊 FINAL EVIDENCE PIPELINE SUMMARY:")
        print(f"  ├─ Evidence queries: {len(evidence_queries)}")
        print(f"  ├─ Total initial results: {sum(r['results'] for r in query_results)}")
        print(f"  ├─ After deduplication: {len(all_chunks)}")
        print(f"  ├─ After reranking: {len(final_passages)}")
        print(f"  └─ Final evidence pieces: {len(final_passages)}")
        
        if final_passages:
            print(f"  📋 FINAL SELECTED EVIDENCE:")
            for i, (sim, rerank) in enumerate(zip(final_similarities, final_rerank_scores)):
                print(f"     #{i+1}: Similarity {sim:.3f}, Rerank {rerank:.3f}")
        
        combined_text = "\n\n".join(final_passages)
        
        # STEP 5: Return comprehensive evidence data
        evidence_data = {
            'passages': final_passages,
            'sources': final_sources,
            'rerank_scores': final_rerank_scores,
            'combined_text': combined_text,
            'similarity_scores': final_similarities,
            'evidence_queries': evidence_queries,
            'vector_distances': [all_chunks[hash(p[:200])]['distance_score'] 
                               for p in final_passages],
            'query_results': query_results,
            'deduplication_stats': {
                'total_chunks_retrieved': sum(r['results'] for r in query_results),
                'unique_chunks': len(all_chunks),
                'final_selected': len(final_passages)
            }
        }
        
        return evidence_data
        
    except Exception as e:
        print(f"  💥 Evidence query generation failed: {e}")
        sim_content = get_category_simulation(requirement_text, category)
        return {
            "passages": [sim_content],
            "sources": ["Simulated Documentation"],
            "similarity_scores": [0.8],
            "rerank_scores": [0.8],
            "combined_text": sim_content,
            "evidence_queries": ["Fallback search"]
        }

async def perform_specialist_evidence_preanalysis(requirement_dict: Dict[str, str], category: str, evidence_data: Dict[str, Any]) -> str:
    """NEW: Specialist pre-analyzes evidence quality before final analysis"""
    print(f"  🧪 EVIDENCE PRE-ANALYSIS by {category.upper()} specialist")
    print(f"  ═══════════════════════════════════════════════════════")
    
    preanalysis_config = EVIDENCE_PREANALYSIS_PROMPTS.get(category)
    if not preanalysis_config:
        print(f"  ⚠️  No evidence pre-analysis configured for {category}")
        return "No evidence pre-analysis available for this category."
    
    model = granite_llm if preanalysis_config["model"] == "granite" else llama_llm
    print(f"  🤖 Using {preanalysis_config['model'].upper()} for evidence assessment")
    
    try:
        print(f"  💭 Specialist analyzing evidence quality...")
        
        preanalysis = (preanalysis_config["prompt"] | model).invoke({
            "requirement": requirement_dict['text'],
            "priority": requirement_dict['priority'],
            "evidence_content": evidence_data['combined_text'],
            "evidence_sources": "\n".join([f"- {source}" for source in evidence_data['sources']]),
            "search_queries": "\n".join([f"- {q}" for q in evidence_data['evidence_queries']])
        })
        
        print(f"  ✅ Evidence pre-analysis completed ({len(preanalysis)} characters)")
        print(f"  📄 Preview: {preanalysis[:200]}...")
        
        return preanalysis
        
    except Exception as e:
        print(f"  💥 Evidence pre-analysis failed: {e}")
        return f"Evidence pre-analysis failed: {str(e)}"

async def analyze_single_requirement_enhanced(requirement_dict: Dict[str, str], category: str, collection_name: str = None) -> Dict[str, Any]:
    """Complete enhanced specialist-driven analysis with evidence pre-analysis"""
    requirement_text = requirement_dict['text']
    priority = requirement_dict['priority']
    
    print(f"\n{'═'*80}")
    print(f"🔬 ENHANCED ANALYSIS: {category.upper()} REQUIREMENT")
    print(f"{'═'*80}")
    print(f"📋 Requirement: {requirement_text}")
    print(f"🎯 Priority: {priority}")
    if collection_name:
        print(f"📚 Collection: {collection_name}")
    print(f"{'═'*80}")
    
    try:
        # PHASE 1: Evidence gathering
        print(f"\n🔍 PHASE 1: EVIDENCE GATHERING")
        evidence_data = await get_evidence_with_specialist_queries(requirement_dict, category, collection_name)
        
        # PHASE 2: NEW - Specialist evidence pre-analysis
        print(f"\n🧪 PHASE 2: EVIDENCE PRE-ANALYSIS")
        evidence_preanalysis = await perform_specialist_evidence_preanalysis(requirement_dict, category, evidence_data)
        
        # PHASE 3: Final analysis with evidence pre-analysis
        print(f"\n🧠 PHASE 3: FINAL SPECIALIST ANALYSIS")
        print(f"══════════════════════════════════════════")
        analysis_config = ANALYSIS_PROMPTS.get(category)
        if not analysis_config:
            return {
                "requirement": requirement_text,
                "category": category,
                "priority": priority,
                "error": f"No analysis configuration for category: {category}"
            }
        
        model = granite_llm if analysis_config["model"] == "granite" else llama_llm
        print(f"🤖 Using {analysis_config['model'].upper()} for final analysis")
        
        # Run final analysis with evidence pre-analysis
        analysis = (analysis_config["prompt"] | model).invoke({
            "requirement": requirement_text,
            "priority": priority,
            "evidence_preanalysis": evidence_preanalysis,
            "sources": "\n".join([f"- {source}" for source in evidence_data['sources']]),
            "retrieved_info": evidence_data['combined_text']
        })
        
        print(f"✅ Final analysis completed ({len(analysis)} characters)")
        
        # Extract scores
        scores = extract_scores_with_support(analysis)
        
        result = {
            "requirement": requirement_text,
            "category": category,
            "priority": priority,
            "analysis": analysis,
            "evidence_preanalysis": evidence_preanalysis,
            "feasibility_score": scores.get('feasibility'),
            "risk_score": scores.get('risk'),
            "support_grade": scores.get('support_grade'),
            "sources": evidence_data['sources'],
            "similarity_scores": evidence_data['similarity_scores'],
            "rerank_scores": evidence_data['rerank_scores'],
            "evidence_queries": evidence_data['evidence_queries'],
            "query_results": evidence_data['query_results'],
            "total_unique_chunks": evidence_data['deduplication_stats']['unique_chunks']
        }
        
        print(f"\n📈 FINAL SCORES:")
        print(f"├─ Feasibility: {result.get('feasibility_score')}")
        print(f"├─ Risk: {result.get('risk_score')}")
        print(f"└─ Evidence Support: {result.get('support_grade')}")
        
        return result
        
    except Exception as e:
        print(f"💥 Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "requirement": requirement_text,
            "category": category,
            "priority": priority,
            "error": f"Analysis failed: {str(e)}"
        }

# STANDALONE TEST REQUIREMENTS FOR DEVELOPMENT
test_requirements_enhanced = {
    "security": [
        {"text": "All user authentication must use multi-factor authentication with TOTP tokens", "priority": "obligatory"},
        {"text": "Database connections must use SSL/TLS encryption with certificate validation", "priority": "obligatory"}
    ],
    "performance": [
        {"text": "System must handle 1000 concurrent users with response times under 2 seconds", "priority": "obligatory"}
    ],
    "functional": [
        {"text": "Users must be able to create custom reports with filtering and sorting", "priority": "obligatory"},
        {"text": "System must provide real-time data synchronization across modules", "priority": "desired"}
    ]
}

# NEW: Test requirements that should NOT be well supported (to test STRONG_NO logic)
test_requirements_unsupported = {
    "security": [
        {"text": "System must integrate with quantum encryption hardware modules", "priority": "obligatory"},
        {"text": "All data must be stored using homomorphic encryption algorithms", "priority": "desired"}
    ],
    "performance": [
        {"text": "System must process 1 million concurrent blockchain transactions per second", "priority": "obligatory"}
    ],
    "functional": [
        {"text": "System must provide AI-powered holographic user interfaces", "priority": "desired"},
        {"text": "Platform must support direct neural interface connectivity", "priority": "optional"}
    ]
}

# TEST SET 1: 3 Requirements from different categories
test_requirements_set1 = {
    "security": [
        {"text": "All data at rest must be encrypted using AES-256 with key rotation every 90 days", "priority": "obligatory"}
    ],
    "performance": [
        {"text": "System must handle 10,000 concurrent active users with peak load of 15,000 users", "priority": "obligatory"}
    ],
    "functional": [
        {"text": "Platform must provide drag-and-drop report designer with 50+ visualization types", "priority": "obligatory"}
    ]
}

# TEST SET 2: Different mix of 3 requirements 
test_requirements_set2 = {
    "security": [
        {"text": "Multi-factor authentication using TOTP with biometric fallback support", "priority": "obligatory"}
    ],
    "performance": [
        {"text": "Auto-scaling infrastructure based on CPU and memory utilization metrics", "priority": "obligatory"}
    ],
    "functional": [
        {"text": "Real-time dashboard updates with WebSocket connections and live data streaming", "priority": "obligatory"}
    ]
}

# TEST SET 3: Mixed results - some supported, some not
test_requirements_set3 = {
    "security": [
        {"text": "System must integrate with quantum encryption hardware modules for data protection", "priority": "obligatory"}
    ],
    "performance": [
        {"text": "Auto-scaling infrastructure based on CPU and memory utilization metrics", "priority": "obligatory"}
    ],
    "functional": [
        {"text": "Platform must support blockchain-based smart contracts for workflow automation", "priority": "obligatory"}
    ]
}

# ============================================================================
# ENHANCED FINAL REPORT GENERATION SYSTEM
# ============================================================================

async def generate_enhanced_category_summary(category: str, requirements_list: List[Dict[str, str]], collection_name: str = None) -> Dict[str, Any]:
    """Generate enhanced category analysis with detailed evidence metrics"""
    if not requirements_list:
        return None
    
    print(f"\n🎯 ANALYZING {category.upper()} CATEGORY ({len(requirements_list)} requirements)")
    print(f"{'='*60}")
    
    # Analyze all requirements in this category
    category_results = []
    for req_idx, requirement in enumerate(requirements_list, 1):
        print(f"\n📋 Requirement {req_idx}/{len(requirements_list)} in {category.upper()}")
        result = await analyze_single_requirement_enhanced(requirement, category, collection_name)
        category_results.append(result)
        
        # Show summary
        if "error" not in result:
            print(f"✅ Complete - Feasibility: {result.get('feasibility_score')}, Support: {result.get('support_grade')}")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    # Calculate category metrics
    successful_results = [r for r in category_results if "error" not in r]
    failed_results = [r for r in category_results if "error" in r]
    
    category_metrics = {
        'category': category,
        'total_requirements': len(requirements_list),
        'successful_analyses': len(successful_results),
        'failed_analyses': len(failed_results),
        'requirements': category_results
    }
    
    if successful_results:
        # Evidence pipeline metrics
        total_evidence_pieces = sum(len(r.get('sources', [])) for r in successful_results)
        total_unique_chunks = sum(r.get('total_unique_chunks', 0) for r in successful_results)
        avg_rerank_score = sum(max(r.get('rerank_scores', [0])) for r in successful_results) / len(successful_results)
        
        # Score distribution
        feasibility_scores = [r.get('feasibility_score') for r in successful_results if r.get('feasibility_score') is not None]
        risk_scores = [r.get('risk_score') for r in successful_results if r.get('risk_score') is not None]
        support_grades = [r.get('support_grade') for r in successful_results if r.get('support_grade')]
        
        category_metrics.update({
            'evidence_pipeline_metrics': {
                'total_evidence_pieces': total_evidence_pieces,
                'total_unique_chunks': total_unique_chunks,
                'average_rerank_score': avg_rerank_score,
                'evidence_per_requirement': total_evidence_pieces / len(successful_results) if successful_results else 0
            },
            'score_metrics': {
                'avg_feasibility': sum(feasibility_scores) / len(feasibility_scores) if feasibility_scores else None,
                'avg_risk': sum(risk_scores) / len(risk_scores) if risk_scores else None,
                'feasibility_range': f"{min(feasibility_scores):.2f} - {max(feasibility_scores):.2f}" if feasibility_scores else "N/A",
                'risk_range': f"{min(risk_scores):.2f} - {max(risk_scores):.2f}" if risk_scores else "N/A",
                'support_distribution': {grade: support_grades.count(grade) for grade in set(support_grades)} if support_grades else {}
            }
        })
    
    print(f"\n📊 CATEGORY SUMMARY for {category.upper()}:")
    print(f"├─ Requirements processed: {len(successful_results)}/{len(requirements_list)}")
    if successful_results:
        print(f"├─ Evidence pieces collected: {category_metrics['evidence_pipeline_metrics']['total_evidence_pieces']}")
        print(f"├─ Avg feasibility score: {category_metrics['score_metrics']['avg_feasibility']:.2f}" if category_metrics['score_metrics']['avg_feasibility'] else "├─ No feasibility scores")
        print(f"└─ Avg rerank score: {category_metrics['evidence_pipeline_metrics']['average_rerank_score']:.2f}")
    
    return category_metrics

# ENHANCED FINAL REPORT PROMPT
enhanced_final_report_llama = PromptTemplate.from_template(
    """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a senior business consultant providing clear, direct answers to stakeholders. Give simple, actionable responses without making users dig through documentation. Focus on what they asked for, what you found, and what they should do.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Create a clear, direct requirements analysis report:

**Requirements Analyzed:**
{category_assessments}

**Project Details:**
- Total Requirements: {total_requirements}
- Categories: {categories_analyzed}
- Documentation Source: {collection_name}

Generate a simple, clear report:

# 📋 Requirements Analysis Report

## 🎯 Executive Summary

**Our Recommendation:** [PROCEED / PROCEED WITH CAUTION / DO NOT PROCEED]

**Bottom Line:** [One sentence summary of project viability]

**Timeline:** [X-Y months for implementation]

**Confidence Level:** [High/Medium/Low] - [Brief reason why]

---

## 📝 What You Asked For vs. What We Found

[For each requirement, create a clear table format:]

| Your Requirement | Evidence Found? | Our Assessment | Recommendation |
|------------------|-----------------|----------------|----------------|
| [Requirement text] | ✅ Strong evidence / ⚠️ Some evidence / ❌ No evidence | Ready to build / Needs research / High risk | Proceed / Plan more / Reconsider |

**Key Findings:**
- **✅ Ready to Build:** [X requirements] - Strong evidence, low risk
- **⚠️ Needs Planning:** [X requirements] - Some gaps, need more research  
- **❌ High Risk:** [X requirements] - No evidence or major concerns

---

## 🚦 What To Do Next

### ✅ Start Immediately (Next 30 days)
**These requirements are ready for development:**
- [List specific requirements]
- **Why:** Strong evidence found, low technical risk
- **Team needed:** [Specific skills]
- **Timeline:** [Specific timeline]

### ⚠️ Plan First (1-3 months planning)
**These need more research before building:**
- [List specific requirements]  
- **Issue:** [Specific problem found]
- **Next step:** [Specific action needed]
- **Decision needed by:** [Date]

### ❌ Reconsider (Major concerns)
**These requirements have serious issues:**
- [List specific requirements]
- **Problem:** [Clear explanation of issue]
- **Options:** [Alternative approaches]
- **Executive decision needed:** [What stakeholders must decide]

---

## 💰 Resources & Timeline

**Development Team Needed:**
- [Specific roles and skills required]

**Realistic Timeline:**
- **Month 1-2:** [Specific deliverables]
- **Month 3-6:** [Specific deliverables]  
- **Month 6+:** [Specific deliverables]

**Budget Considerations:**
- [Key cost factors identified]
- [Potential cost risks]

---

## ⚠️ Key Risks & Mitigations

**Biggest Risks:**
1. **[Risk 1]:** [Simple explanation] → **Solution:** [Clear mitigation]
2. **[Risk 2]:** [Simple explanation] → **Solution:** [Clear mitigation]

**Red Flags:**
- [Any requirements that should not be built as specified]
- [Any technical impossibilities found]

---

## 🎯 Decision Points

**Immediate Decisions Needed:**
- [Specific decisions with deadlines]

**Future Decisions:**
- [Decisions that can wait but have deadlines]

**Success Metrics:**
- [How to measure if this project succeeds]

---

**Instructions for the AI:**
- Be direct and specific, no fluff
- If no evidence was found for a requirement, say so clearly
- Give actual timelines, not vague ranges
- Identify specific skills needed
- Point out any requirements that are technically impossible
- Use simple language that executives can understand immediately
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
)

async def generate_enhanced_final_report(all_results: List[Dict[str, Any]], collection_name: str = None) -> str:
    """Generate comprehensive enhanced final report with business-focused insights"""
    print("\n🏗️  GENERATING ENHANCED FINAL REPORT")
    print("="*60)
    
    # Filter and organize results
    successful_categories = [r for r in all_results if r is not None and 'requirements' in r]
    
    if not successful_categories:
        return "❌ **Error**: No valid analysis results to generate report."
    
    # Calculate overall metrics
    total_requirements = sum(cat['total_requirements'] for cat in successful_categories)
    total_successful = sum(cat['successful_analyses'] for cat in successful_categories)
    
    categories_analyzed = [cat['category'] for cat in successful_categories]
    
    print(f"📊 Report Statistics:")
    print(f"├─ Requirements analyzed: {total_successful}/{total_requirements}")
    print(f"├─ Categories covered: {len(categories_analyzed)}")
    print(f"└─ Collection used: {collection_name or 'Default'}")
    
    # Build business-focused category assessments
    business_assessments = []
    
    for category_data in successful_categories:
        category = category_data['category']
        print(f"\n📋 Processing {category.upper()} category report section...")
        
        # Build detailed requirement-by-requirement assessment
        category_section = f"**{category.upper()} REQUIREMENTS:**\n\n"
        
        # Add individual requirements with clear evidence and assessment
        for req_data in category_data['requirements']:
            if 'error' in req_data:
                category_section += f"❌ **{req_data['requirement']}**\n"
                category_section += f"   - ERROR: Analysis failed\n\n"
                continue
            
            # Get the requirement details
            requirement_text = req_data.get('requirement', 'Unknown requirement')
            feasibility = req_data.get('feasibility_score', 0)
            support = req_data.get('support_grade', 'UNKNOWN')
            analysis = req_data.get('analysis', '')
            evidence_preanalysis = req_data.get('evidence_preanalysis', '')
            
            # Determine evidence strength from the analysis
            evidence_found = "Strong evidence"
            if support in ['STRONG_YES']:
                evidence_found = "✅ Strong evidence"
            elif support in ['WEAK_YES', 'UNSURE']:
                evidence_found = "⚠️ Some evidence"
            else:
                evidence_found = "❌ No evidence"
            
            # Simple assessment
            if feasibility >= 0.8:
                assessment = "Ready to build"
                recommendation = "Proceed"
            elif feasibility >= 0.6:
                assessment = "Needs research"
                recommendation = "Plan more"
            else:
                assessment = "High risk"
                recommendation = "Reconsider"
            
            # Extract key insights from specialist analysis
            evidence_comment = ""
            if evidence_preanalysis:
                # Look for evidence quality assessment
                if "EVIDENCE QUALITY: EXCELLENT" in evidence_preanalysis:
                    evidence_comment = "Comprehensive documentation found supporting this requirement."
                elif "EVIDENCE QUALITY: GOOD" in evidence_preanalysis:
                    evidence_comment = "Good documentation found with some implementation details."
                elif "EVIDENCE QUALITY: FAIR" in evidence_preanalysis:
                    evidence_comment = "Limited documentation found, some gaps in coverage."
                elif "EVIDENCE QUALITY: POOR" in evidence_preanalysis:
                    evidence_comment = "Little to no documentation found supporting this requirement."
                else:
                    evidence_comment = "Documentation review completed."
            
            # Add structured requirement data
            category_section += f"**Requirement:** {requirement_text}\n"
            category_section += f"**Evidence Found:** {evidence_found}\n"
            category_section += f"**Assessment:** {assessment}\n"
            category_section += f"**Recommendation:** {recommendation}\n"
            if evidence_comment:
                category_section += f"**Evidence Notes:** {evidence_comment}\n"
            category_section += "\n"
        
        business_assessments.append(category_section)
    
    # Build business-focused evidence summary (without technical details)
    evidence_summary = f"""
**Analysis Quality Overview:**
- Requirements analyzed across {len(categories_analyzed)} business areas
- Documentation review completed for {collection_name or 'available resources'}
- Analysis confidence based on available supporting documentation
- Recommendations provided for each requirement category
"""
    
    all_category_assessments = "\n".join(business_assessments)
    
    print(f"📝 Invoking enhanced report generation...")
    
    try:
        enhanced_report = (enhanced_final_report_llama | llama_llm).invoke({
            "total_requirements": total_requirements,
            "categories_analyzed": ", ".join(categories_analyzed),
            "collection_name": collection_name or "Available Documentation",
            "category_assessments": all_category_assessments,
            "evidence_metrics": evidence_summary
        })
        
        print(f"✅ Enhanced report generated ({len(enhanced_report)} characters)")
        return enhanced_report
        
    except Exception as e:
        print(f"❌ Enhanced report generation failed: {e}")
        return f"**Report Generation Error**: {str(e)}\n\nFallback summary: Analyzed {total_successful} requirements across {len(categories_analyzed)} categories."

async def test_unsupported_requirements():
    """Test requirements that should NOT be well supported to verify STRONG_NO detection"""
    print("="*80)
    print("🧪 TESTING UNSUPPORTED REQUIREMENTS (Should get WEAK_NO/STRONG_NO)")
    print("="*80)
    
    test_collection = "test_product_docs"
    print(f"🎯 Testing with collection: {test_collection}")
    print(f"📄 Document: EnterpriseFlow Platform (should NOT support these requirements)")
    
    # Test one requirement from each category that should fail
    test_cases = [
        {"text": "System must integrate with quantum encryption hardware modules", "priority": "obligatory", "category": "security"},
        {"text": "System must process 1 million concurrent blockchain transactions per second", "priority": "obligatory", "category": "performance"},
        {"text": "System must provide AI-powered holographic user interfaces", "priority": "desired", "category": "functional"}
    ]
    
    for test_case in test_cases:
        print(f"\n🔬 Testing {test_case['category'].upper()}: {test_case['text'][:60]}...")
        result = await analyze_single_requirement_enhanced(test_case, test_case['category'], test_collection)
        
        if "error" not in result:
            support = result.get('support_grade', 'UNKNOWN')
            feasibility = result.get('feasibility_score', 'N/A')
            print(f"📊 Result: Support={support}, Feasibility={feasibility}")
            
            if support in ['STRONG_NO', 'WEAK_NO']:
                print(f"✅ CORRECT: System properly detected unsupported requirement")
            elif support in ['UNSURE']:
                print(f"🟡 PARTIAL: System detected uncertainty (acceptable)")
            else:
                print(f"❌ ISSUE: System gave {support} for unsupported requirement")
        else:
            print(f"💥 Error: {result.get('error')}")

async def test_set1_three_requirements():
    """Test Set 1: 3 requirements from different categories"""
    print("="*80)
    print("🧪 TEST SET 1: 3 Requirements from Different Categories")
    print("="*80)
    
    test_collection = "test_product_docs"
    print(f"🎯 Testing with collection: {test_collection}")
    
    # Process all categories
    all_category_results = []
    
    for category in ["security", "performance", "functional"]:
        requirements = test_requirements_set1.get(category, [])
        if not requirements:
            continue
            
        category_result = await generate_enhanced_category_summary(category, requirements, test_collection)
        if category_result:
            all_category_results.append(category_result)
    
    # Generate final report
    final_report = await generate_enhanced_final_report(all_category_results, test_collection)
    
    print("\n" + "="*80)
    print("🎯 TEST SET 1 FINAL REPORT")
    print("="*80)
    print(final_report)
    print("="*80)
    
    return final_report

async def test_set2_three_requirements():
    """Test Set 2: Different mix of 3 requirements"""
    print("="*80)
    print("🧪 TEST SET 2: Different Mix of 3 Requirements")
    print("="*80)
    
    test_collection = "test_product_docs"
    print(f"🎯 Testing with collection: {test_collection}")
    
    # Process all categories
    all_category_results = []
    
    for category in ["security", "performance", "functional"]:
        requirements = test_requirements_set2.get(category, [])
        if not requirements:
            continue
            
        category_result = await generate_enhanced_category_summary(category, requirements, test_collection)
        if category_result:
            all_category_results.append(category_result)
    
    # Generate final report
    final_report = await generate_enhanced_final_report(all_category_results, test_collection)
    
    print("\n" + "="*80)
    print("🎯 TEST SET 2 FINAL REPORT")
    print("="*80)
    print(final_report)
    print("="*80)
    
    return final_report

async def test_set3_mixed_requirements():
    """Test Set 3: Mixed requirements - some should be supported, some not"""
    print("="*80)
    print("🧪 TEST SET 3: Mixed Requirements (Some Supported, Some Not)")
    print("="*80)
    
    test_collection = "test_product_docs"
    print(f"🎯 Testing with collection: {test_collection}")
    print("📋 Expected: Auto-scaling=STRONG_YES, Quantum=WEAK_NO, Blockchain=WEAK_NO")
    
    # Process all categories
    all_category_results = []
    
    for category in ["security", "performance", "functional"]:
        requirements = test_requirements_set3.get(category, [])
        if not requirements:
            continue
            
        category_result = await generate_enhanced_category_summary(category, requirements, test_collection)
        if category_result:
            all_category_results.append(category_result)
    
    # Generate final report
    final_report = await generate_enhanced_final_report(all_category_results, test_collection)
    
    print("\n" + "="*80)
    print("🎯 TEST SET 3 FINAL REPORT")
    print("="*80)
    print(final_report)
    print("="*80)
    
    return final_report

# ============================================================================
# API COMPATIBILITY WRAPPERS FOR EXISTING SERVER
# ============================================================================

async def analyze_category_requirements_specialist(category: str, requirements_list: List[Dict[str, str]], collection_name: str = None) -> Dict[str, Any]:
    """
    API compatibility wrapper for existing server.
    
    Takes a category and list of requirements in the format:
    [{"text": "requirement text", "priority": "obligatory"}, ...]
    
    Returns the enhanced category summary compatible with the existing API.
    """
    print(f"\n🔄 API WRAPPER: analyze_category_requirements_specialist")
    print(f"├─ Category: {category}")
    print(f"├─ Requirements: {len(requirements_list)}")
    print(f"└─ Collection: {collection_name or 'default'}")
    
    try:
        # Use the enhanced category summary function
        result = await generate_enhanced_category_summary(category, requirements_list, collection_name)
        print(f"✅ API wrapper completed successfully for {category}")
        return result
        
    except Exception as e:
        print(f"❌ API wrapper failed for {category}: {e}")
        return {
            'category': category,
            'total_requirements': len(requirements_list),
            'successful_analyses': 0,
            'failed_analyses': len(requirements_list),
            'error': f"Category analysis failed: {str(e)}",
            'requirements': []
        }

async def generate_final_specialist_report(category_results: List[Dict[str, Any]], collection_name: str = None) -> str:
    """
    API compatibility wrapper for existing server.
    
    Takes category results and returns final report string.
    """
    print(f"\n📝 API WRAPPER: generate_final_specialist_report")
    print(f"├─ Categories: {len(category_results)}")
    print(f"└─ Collection: {collection_name or 'auto-detected'}")
    
    try:
        # Use the enhanced final report function
        report = await generate_enhanced_final_report(category_results, collection_name)
        print(f"✅ API wrapper generated final report ({len(report)} characters)")
        return report
        
    except Exception as e:
        print(f"❌ API wrapper final report failed: {e}")
        # Fallback simple report
        successful_categories = [r for r in category_results if r and 'error' not in r]
        total_requirements = sum(r.get('total_requirements', 0) for r in category_results if r)
        
        return f"""# Requirements Analysis Report

## Summary
- **Total Requirements Analyzed**: {total_requirements}
- **Categories**: {len(category_results)}
- **Status**: Analysis completed with some limitations

## Error
The enhanced report generation encountered an error: {str(e)}

## Basic Results
{chr(10).join([f"- **{r.get('category', 'Unknown')}**: {r.get('successful_analyses', 0)}/{r.get('total_requirements', 0)} requirements analyzed" for r in category_results if r])}

Please check the server logs for detailed error information.
"""

def test_model_connections() -> bool:
    """
    API compatibility wrapper for model connection testing.
    
    Tests both LLM models and reranker connectivity.
    """
    print(f"\n🔍 API WRAPPER: test_model_connections")
    
    try:
        # Test granite LLM
        print("├─ Testing Granite LLM...")
        granite_test = granite_llm.invoke("Test connection")
        print(f"│  ✅ Granite response: {len(granite_test)} characters")
        
        # Test llama LLM  
        print("├─ Testing Llama LLM...")
        llama_test = llama_llm.invoke("Test connection")
        print(f"│  ✅ Llama response: {len(llama_test)} characters")
        
        # Test reranker
        print("├─ Testing Reranker...")
        if reranker:
            print("│  ✅ Reranker initialized")
        else:
            print("│  ⚠️  Reranker not available")
        
        # Test database
        print("└─ Testing Database...")
        if db:
            doc_count = db._collection.count()
            print(f"   ✅ Database connected ({doc_count} documents)")
        else:
            print(f"   ⚠️  Database not available")
        
        # All core components working
        core_working = bool(granite_test and llama_test and db)
        print(f"🎯 Model connections test: {'✅ PASSED' if core_working else '⚠️  PARTIAL'}")
        return core_working
        
    except Exception as e:
        print(f"❌ Model connection test failed: {e}")
        return False

async def analyze_single_requirement_specialist(requirement_dict: Dict[str, str], category: str, collection_name: str = None) -> Dict[str, Any]:
    """
    API compatibility wrapper for single requirement analysis.
    
    Takes a requirement dict in format: {"text": "...", "priority": "..."}
    Returns analysis result compatible with existing API.
    """
    print(f"\n🔄 API WRAPPER: analyze_single_requirement_specialist")
    print(f"├─ Category: {category}")
    print(f"├─ Requirement: {requirement_dict.get('text', '')[:60]}...")
    print(f"└─ Collection: {collection_name or 'default'}")
    
    try:
        # Use the enhanced single requirement function
        result = await analyze_single_requirement_enhanced(requirement_dict, category, collection_name)
        print(f"✅ API wrapper single analysis completed")
        return result
        
    except Exception as e:
        print(f"❌ API wrapper single analysis failed: {e}")
        return {
            "requirement": requirement_dict.get('text', ''),
            "category": category,
            "priority": requirement_dict.get('priority', 'unknown'),
            "error": f"Single requirement analysis failed: {str(e)}"
        }

if __name__ == "__main__":
    print("Starting TEST SET 3...")
    asyncio.run(test_set3_mixed_requirements()) 