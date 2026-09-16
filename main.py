#!/usr/bin/env python3
import os,sys,re,json,time,hashlib,pickle,math,csv
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

CPU=os.cpu_count() or 4
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k,str(CPU))

MAX_VOCAB=16000; MIN_FREQ=1; TOP_K=8; CHUNK_SIZE=1800; OVERLAP=250
EMBEDDING_MODEL=os.getenv("ORIN_EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE=int(os.getenv("ORIN_EMBEDDING_BATCH_SIZE","32"))
MODELS=Path("models"); MODELS.mkdir(exist_ok=True)
INDEX_FILE=MODELS/"orin_retrieval.pkl"; MEMORY_FILE=MODELS/"orin_memory.json"
TREE_FILE=MODELS/"knowledge_tree.json"; META_FILE=MODELS/"orin_meta.json"
PAIR_RE=re.compile(r"^\s*(User|Assistant)\s*:\s*(.*)$",re.I)
TOKEN_RE=re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?",re.UNICODE)

def char_ngrams(s,n=3):
    s=" "+norm(s)+" "
    if len(s)<n: return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}

def semantic_similarity(a,b):
    aa=char_ngrams(a); bb=char_ngrams(b)
    if not aa or not bb: return 0.0
    return len(aa&bb)/math.sqrt(len(aa)*len(bb))

def definition_subject(q):
    m=re.match(r"^(what is|what are|define|definition of) (.+?)\??$",norm(q))
    return m.group(2).strip() if m else ""

def norm(s): return " ".join(TOKEN_RE.findall(str(s).lower()))
def sha(s): return hashlib.sha256(s.encode("utf-8",errors="ignore")).hexdigest()

class TransformerSimilarity:
    """Optional sentence-transformer backend with a lexical fallback."""
    def __init__(self,model_name=EMBEDDING_MODEL):
        self.model_name=model_name
        self.model=None
        self.failed=False

    def _load(self):
        if self.model is not None or self.failed: return self.model
        try:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(self.model_name)
            print(f"[EMBEDDINGS] Loaded {self.model_name}")
        except Exception as e:
            self.failed=True
            print(f"[WARN] Transformer similarity unavailable: {e}")
        return self.model

    def encode(self,texts):
        model=self._load()
        if model is None or not texts: return None
        try:
            return np.asarray(model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),dtype=np.float32)
        except Exception as e:
            self.failed=True
            print(f"[WARN] Transformer encoding failed: {e}")
            return None

def clean_pairs(text):
    out=[]; q=None
    for line in text.splitlines():
        m=PAIR_RE.match(line)
        if not m: continue
        role,val=m.group(1).lower(),m.group(2).strip()
        if role=="user": q=val
        elif q and val: out.append((q,val)); q=None
    return out

def squad_pairs(path):
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    out=[]
    for article in obj.get("data",[]):
        for para in article.get("paragraphs",[]):
            context=str(para.get("context","")).strip()
            for qa in para.get("qas",[]):
                q=str(qa.get("question","")).strip()
                answers=qa.get("answers") or []
                if q and answers:
                    a=str(answers[0].get("text","")).strip()
                    if a: out.append((q,a+" Context: "+context))
    return out

def json_text(x):
    if isinstance(x,dict): return "\n".join(f"{k}: {json_text(v)}" for k,v in x.items())
    if isinstance(x,list): return "\n".join(json_text(v) for v in x)
    return str(x)

def read_file(p):
    ext=p.suffix.lower()
    if ext==".json":
        try:
            obj=json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj,dict) and "data" in obj and isinstance(obj["data"],list):
                pairs=squad_pairs(p)
                if pairs: return "pairs",pairs
            return "doc",json_text(obj)
        except Exception: return "doc",p.read_text(encoding="utf-8",errors="ignore")
    if ext==".csv":
        with p.open(newline="",encoding="utf-8",errors="ignore") as f: return "doc","\n".join(" | ".join(x.strip() for x in r if x.strip()) for r in csv.reader(f))
    if ext==".pdf":
        try:
            import pypdf
            r=pypdf.PdfReader(str(p)); return "doc","\n".join(pg.extract_text() or "" for pg in r.pages)
        except Exception as e: print(f"[WARN] PDF skipped: {p} ({e})"); return "doc",""
    if ext==".docx":
        try:
            from docx import Document
            d=Document(str(p)); return "doc","\n".join(x.text for x in d.paragraphs)
        except Exception as e: print(f"[WARN] DOCX skipped: {p} ({e})"); return "doc",""
    return "doc",p.read_text(encoding="utf-8",errors="ignore")

def discover(paths):
    exts={".txt",".md",".rst",".log",".text",".json",".csv",".pdf",".docx"}; seen=set()
    for raw in paths:
        p=Path(raw)
        if not p.exists(): print(f"[WARN] Missing: {p}"); continue
        fs=[p] if p.is_file() else [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts]
        for f in fs:
            f=f.resolve()
            if str(f) not in seen: seen.add(str(f)); yield f

def chunks(text):
    paras=[x.strip() for x in re.split(r"\n\s*\n",text) if x.strip()]; out=[]
    for s in paras:
        if len(s)<=CHUNK_SIZE: out.append(s); continue
        start=0
        while start<len(s):
            end=min(len(s),start+CHUNK_SIZE); out.append(s[start:end])
            if end==len(s): break
            start=max(0,end-OVERLAP)
    return out

class Index:
    def __init__(self):
        self.docs=[]
        self.postings=defaultdict(dict)
        self.df=Counter()
        self.N=0
        self.hashes=set()
        self.embedding_model_name=EMBEDDING_MODEL
        self.embeddings=None
        self.embedding_hashes=[]
        self._embedder=TransformerSimilarity(self.embedding_model_name)

    def add(self,text,source,kind="document",question=None,answer=None):
        text=text.strip()
        if not text: return False

        # Defensive initialization for indexes created by older versions.
        if not hasattr(self,"hashes"):
            self.hashes=set()
            for d in getattr(self,"docs",[]):
                old_hash=d.get("hash")
                if old_hash:
                    self.hashes.add(old_hash)

        h=sha(kind+"|"+str(source)+"|"+text)
        if h in self.hashes: return False

        i=len(self.docs)
        self.docs.append({
            "hash":h,"text":text,"source":str(source),"kind":kind,
            "question":question,"answer":answer
        })
        self.hashes.add(h)
        return True

    def rebuild(self):
        self.postings=defaultdict(dict)
        self.df=Counter()
        self.N=len(self.docs)
        self.hashes=set()

        for i,d in enumerate(self.docs):
            self.hashes.add(d.get("hash",""))
            terms=norm(d.get("text","")).split()
            counts=Counter(terms)
            for t,tf in counts.items():
                self.postings[t][i]=tf
                self.df[t]+=1

        self._rebuild_embeddings()

    def _rebuild_embeddings(self):
        texts=[self._embedding_text(d) for d in self.docs]
        if not texts:
            self.embeddings=None
            self.embedding_hashes=[]
            return
        embeddings=self._embedder.encode(texts)
        if embeddings is None:
            return
        self.embeddings=embeddings
        self.embedding_hashes=[d.get("hash","") for d in self.docs]

    @staticmethod
    def _embedding_text(doc):
        question=doc.get("question") or ""
        answer=doc.get("answer") or ""
        if question:
            return f"Question: {question} Answer: {answer}"
        return doc.get("text","")

    def _ensure_embeddings(self):
        if (self.embeddings is None or len(self.embedding_hashes)!=len(self.docs)
                or self.embedding_hashes!=[d.get("hash","") for d in self.docs]):
            self._rebuild_embeddings()

    def search(self,q,k=TOP_K):
        nq=norm(q)
        terms=nq.split()
        if not terms or not self.docs: return []

        qset=set(terms)
        scores=defaultdict(float)

        # Semantic retrieval supplies candidates even when a paraphrase has no
        # exact token overlap with the stored question.
        semantic_scores={}
        self._ensure_embeddings()
        if self.embeddings is not None:
            query_embedding=self._embedder.encode([q])
            if query_embedding is not None:
                similarities=self.embeddings @ query_embedding[0]
                candidate_count=min(len(similarities),max(k*8,32))
                for i in np.argpartition(similarities,-candidate_count)[-candidate_count:]:
                    semantic_scores[int(i)]=float(similarities[i])
                    scores[int(i)]+=80*max(0.0,float(similarities[i]))

        # Lexical retrieval remains valuable for exact names, identifiers, and
        # small datasets where embeddings are not available.
        for t in qset:
            post=self.postings.get(t,{})
            if not post: continue
            idf=math.log((self.N+1)/(self.df[t]+1))+1
            for i,tf in post.items():
                scores[i]+=idf*(1+math.log(tf))

        if not scores: return []

        subject=definition_subject(q)

        # Rerank the small candidate set instead of scanning the full database.
        for i in list(scores):
            d=self.docs[i]
            question=norm(d.get("question") or "")
            answer=norm(d.get("answer") or "")
            text=norm(d.get("text") or "")

            if question:
                qt=set(question.split())
                at=set(answer.split())

                qcov=len(qset&qt)/max(1,len(qset))
                acov=len(qset&at)/max(1,len(qset))

                # Question similarity dominates answer-word overlap.
                scores[i]+=35*qcov
                scores[i]+=3*acov

                # Offline character n-gram similarity catches wording variations.
                scores[i]+=45*semantic_similarity(nq,question)

                if i in semantic_scores:
                    scores[i]+=90*semantic_scores[i]

                if question==nq:
                    scores[i]+=1500
                elif nq in question:
                    scores[i]+=120

                # Definition questions must actually discuss the requested subject.
                if subject:
                    if subject in question:
                        scores[i]+=200
                    else:
                        scores[i]*=0.12
            else:
                # Generic documents are fallback knowledge.
                dset=set(text.split())
                scores[i]+=2*len(qset&dset)/max(1,len(qset))
                scores[i]+=10*semantic_similarity(nq,text[:1500])
                if i in semantic_scores:
                    scores[i]+=40*semantic_scores[i]

        return sorted(scores.items(),key=lambda x:x[1],reverse=True)[:k]

    def save(self):
        with open(INDEX_FILE,"wb") as f:
            pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)

def load_index():
    if INDEX_FILE.exists():
        try:
            with open(INDEX_FILE,"rb") as f:
                idx=pickle.load(f)

            # Backward compatibility: v4.3 and earlier indexes did not
            # contain hashes. Reconstruct them from the existing documents.
            if not hasattr(idx,"hashes"):
                idx.hashes=set()
                for d in getattr(idx,"docs",[]):
                    h=d.get("hash")
                    if h:
                        idx.hashes.add(h)

            if not hasattr(idx,"embedding_model_name"):
                idx.embedding_model_name=EMBEDDING_MODEL
            if not hasattr(idx,"embeddings"):
                idx.embeddings=None
            if not hasattr(idx,"embedding_hashes"):
                idx.embedding_hashes=[]
            idx._embedder=TransformerSimilarity(idx.embedding_model_name)

            # Rebuild the inverted index if an older index format is loaded.
            if not hasattr(idx,"postings") or not hasattr(idx,"df"):
                idx.rebuild()

            return idx
        except Exception as e:
            print(f"[INDEX] Could not load old index: {e}")
            print("[INDEX] Starting a fresh index.")
    return Index()

def save_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")

def update_tree(pairs):
    # Rebuild from the persistent memory so an older tree format cannot crash v4.
    tree={}
    for q,a in pairs:
        m=re.match(r"^\s*what\s+is\s+([^:?!]+)(?::\s*(.*?))?\??\s*$",q,re.I)
        if not m: continue
        topic=m.group(1).strip()
        path=[x.strip() for x in (m.group(2) or "").split(",") if x.strip()]
        node=tree.setdefault(topic,{"questions":[],"branches":{}})
        for part in path:
            branch=node["branches"].setdefault(part,{"questions":[],"branches":{}})
            # Repair any malformed/legacy branch loaded from an older tree.
            if not isinstance(branch,dict):
                branch={"questions":[],"branches":{}}
                node["branches"][part]=branch
            branch.setdefault("questions",[])
            branch.setdefault("branches",{})
            node=branch
        node.setdefault("questions",[])
        item={"question":q,"answer":a}
        if item not in node["questions"]:
            node["questions"].append(item)
    save_json(TREE_FILE,tree)
    return len(tree)

def train(paths):
    idx=load_index(); added=0; pairs_all=[]; meta={}
    if META_FILE.exists():
        try: meta=json.loads(META_FILE.read_text())
        except Exception: pass
    for p in discover(paths):
        key=str(p); fp=f"{p.stat().st_size}:{p.stat().st_mtime_ns}"
        if meta.get(key)==fp: print(f"[SKIP] unchanged: {p}"); continue
        kind,data=read_file(p)
        if kind=="pairs":
            seen=set(); ps=[]
            for q,a in data:
                nq=norm(q)
                if nq and nq not in seen: seen.add(nq); ps.append((q,a))
            print(f"[SQuAD] {p}: {len(ps):,} answerable Q&A")
            for q,a in ps:
                if idx.add(f"Question: {q}\nAnswer: {a}",p,"qa",q,a): added+=1; pairs_all.append((q,a))
        else:
            ps=clean_pairs(data)
            if ps:
                seen=set(); print(f"[Q&A] {p}: {len(ps):,}")
                for q,a in ps:
                    nq=norm(q)
                    if nq in seen: continue
                    seen.add(nq)
                    if idx.add(f"Question: {q}\nAnswer: {a}",p,"qa",q,a): added+=1; pairs_all.append((q,a))
            else:
                cs=chunks(data); print(f"[DOC] {p}: {len(cs):,} chunks")
                for c in cs:
                    if idx.add(c,p,"document"): added+=1
        meta[key]=fp
    idx.rebuild(); idx.save()
    mem={}
    if MEMORY_FILE.exists():
        try: mem=json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception: pass
    for q,a in pairs_all: mem[norm(q)]=a
    save_json(MEMORY_FILE,mem)
    all_mem_pairs=list(mem.items())
    topics=update_tree(all_mem_pairs)
    save_json(META_FILE,meta)
    print("\n====================================================================")
    print(f"ORIN READY | Knowledge: {len(idx.docs):,} | New: {added:,} | Memories: {len(mem):,} | Tree topics: {topics:,}")
    print("Existing knowledge preserved. SQuAD Q&A uses question-first + semantic reranking.")

def answer(q):
    idx=load_index()
    mem={}
    if MEMORY_FILE.exists():
        try: mem=json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception: pass

    nq=norm(q)

    # 1. Exact memory = highest confidence.
    if nq in mem:
        return mem[nq]

    # 2. Try "what is X" against a memory stored as just "X".
    subject=definition_subject(q)
    if subject and subject in mem:
        return mem[subject]

    # 3. Fast indexed retrieval + question-aware reranking.
    hits=idx.search(q)
    if not hits:
        return "I don't have enough information in my knowledge base to answer reliably."

    best_i,best_score=hits[0]
    best=idx.docs[best_i]

    # Reject weak keyword matches instead of confidently returning unrelated data.
    if best_score<8:
        return "I don't have enough information in my knowledge base to answer reliably."

    # Q&A result: return the clean answer only.
    if best.get("answer"):
        return best["answer"]

    # Document result: return one strongest chunk as fallback.
    return f"[{best.get('source','unknown')}]\\n{best.get('text','')[:1200]}"

def chat():
    print("ORIN HYBRID v4.4.1 ONLINE | /exit /stats /search <query>")
    while True:
        try: q=input("\nYou: ").strip()
        except (EOFError,KeyboardInterrupt): print(); return
        if q.lower() in {"/exit","exit","quit"}: return
        if q.lower()=="/stats":
            idx=load_index(); print(f"Knowledge: {len(idx.docs):,}"); continue
        if q.lower().startswith("/search "):
            idx=load_index()
            for i,s in idx.search(q[8:],10):
                d=idx.docs[i]; print(f"\n{s:.2f} | {d['source']}\n{d['text'][:500]}")
            continue
        print("Orin:",answer(q))

if __name__=="__main__":
    START=time.time()
    if len(sys.argv)<2: print("Usage: python main.py train <files/folders> | chat"); sys.exit(1)
    if sys.argv[1].lower()=="train":
        if len(sys.argv)<3: print("Usage: python main.py train <files/folders>"); sys.exit(1)
        train(sys.argv[2:])
    elif sys.argv[1].lower()=="chat": chat()
    else: print("Unknown command:",sys.argv[1])
