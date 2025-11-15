        // main.cpp
        // Full C++ demo integrating FAISS for dense recall and implementing hybrid search + feed ranking.
        // Compile with CMake; ensure FAISS C++ library is installed and visible to linker.

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <sstream>
#include <memory>

// Include FAISS header if available
#ifdef __has_include
#  if __has_include(<faiss/IndexFlat.h>)
#    include <faiss/IndexFlat.h>
#    define HAVE_FAISS 1
#  endif
#endif

using namespace std;

float dot(const vector<float>& a, const vector<float>& b) {
    float s=0; size_t n=min(a.size(), b.size());
    for(size_t i=0;i<n;++i) s += a[i]*b[i];
    return s;
}
float norm(const vector<float>& a){ return sqrtf(max(0.0f, dot(a,a))); }
vector<float> normalize(const vector<float>& a){ float n=norm(a)+1e-9f; vector<float> out(a.size()); for(size_t i=0;i<a.size();++i) out[i]=a[i]/n; return out; }
float cosine_sim(const vector<float>& a, const vector<float>& b){ return dot(a,b)/( (norm(a)+1e-9f)*(norm(b)+1e-9f) ); }

struct ASRHyp{ string hypothesis; float score; };
struct Segment{ int id; int video_id; float start,end; string transcript; vector<ASRHyp> asr_lattice; float asr_confidence; };
struct Video{ int id; string title; float created_ts; float duration; vector<int> segment_ids; vector<float> embedding; };

// Simple softmax over negative-shift scores
vector<float> softmax_negshift(const vector<float>& scores){
    float mn = *min_element(scores.begin(), scores.end());
    vector<float> exps(scores.size()); double s=0;
    for(size_t i=0;i<scores.size();++i){ exps[i]=exp(-(scores[i]-mn)); s+=exps[i]; }
    for(size_t i=0;i<exps.size();++i) exps[i]/= (s+1e-12);
    return exps;
}

float expected_token_count(const string& word, const Segment& seg){
    if(seg.asr_lattice.empty()){
        int cnt=0; stringstream ss(seg.transcript); string w; while(ss>>w){ if(w==word) ++cnt; } return cnt;
    }
    vector<float> scores; for(auto &h: seg.asr_lattice) scores.push_back(h.score);
    auto probs = softmax_negshift(scores);
    float total=0; for(size_t i=0;i<probs.size();++i){ stringstream ss(seg.asr_lattice[i].hypothesis); string w; int c=0; while(ss>>w) if(w==word) ++c; total += probs[i]*c; }
    return total;
}

float token_present_prob(const string& word, const Segment& seg){
    if(seg.asr_lattice.empty()){
        return seg.transcript.find(word)!=string::npos ? 1.0f : 0.0f;
    }
    vector<float> scores; for(auto &h: seg.asr_lattice) scores.push_back(h.score);
    auto probs = softmax_negshift(scores);
    float pnot=1.0f; for(size_t i=0;i<probs.size();++i){ if(seg.asr_lattice[i].hypothesis.find(word)!=string::npos) pnot *= (1.0f - probs[i]); }
    return 1.0f - pnot;
}

float bm25_expected_score(const vector<string>& qtokens, const Segment& seg){
    const float k1=1.2f, b=0.75f, avg_len=8.0f; float seg_len = max(1.0f, (float)max(1,(int)count(seg.transcript.begin(), seg.transcript.end(), ' ')+1)); float score=0.0f;
    for(auto &w: qtokens){ float f = expected_token_count(w, seg); float num = f*(k1+1.0f); float den = f + k1*(1.0f - b + b*(seg_len/avg_len)); score += (num / (den + 1e-9f)); }
    return score;
}

// Dataset containers
static vector<Segment> segments;
static vector<Video> videos;

#ifdef HAVE_FAISS
using namespace faiss;
#endif

// FAISS wrapper class
class FaissIndexWrapper{
public:
    FaissIndexWrapper(int dim): dim_(dim){
#ifdef HAVE_FAISS
        idx_ = new IndexFlatIP(dim_);
#endif
    }
    ~FaissIndexWrapper(){
#ifdef HAVE_FAISS
        delete idx_;
#endif
    }
    void add(const vector<vector<float>>& vecs){
#ifdef HAVE_FAISS
        size_t n = vecs.size();
        if(n==0) return;
        vector<float> flat; flat.reserve(n*dim_);
        for(auto &v: vecs){ for(float x: v) flat.push_back(x); }
        idx_->add(n, flat.data());
#endif
    }
    vector<pair<int,float>> search(const vector<float>& q, int topk){
        vector<pair<int,float>> out;
#ifdef HAVE_FAISS
        if(!idx_) return out;
        vector<float> qn = q; float qnorm = 0; for(float x: qn) qnorm += x*x; if(qnorm>0){ float inv = 1.0f/sqrtf(qnorm); for(auto &x: qn) x*=inv; }
        vector<int64_t> labels(topk); vector<float> distances(topk);
        idx_->search(1, qn.data(), topk, distances.data(), labels.data());
        for(int i=0;i<topk;++i){ if(labels[i]<0) break; out.push_back({(int)labels[i], distances[i]}); }
#endif
        return out;
    }
private:
    int dim_;
#ifdef HAVE_FAISS
    IndexFlatIP* idx_ = nullptr;
#endif
};

// dense recall brute-force (fallback if FAISS not present)
vector<pair<int,float>> dense_recall_bruteforce(const vector<float>& qvec, int topn=100){
    vector<pair<int,float>> scored;
    for(auto &v: videos){ float s = cosine_sim(qvec, v.embedding); scored.push_back({v.id, s}); }
    sort(scored.begin(), scored.end(), [](auto &a, auto &b){ return a.second > b.second; }); if((int)scored.size()>topn) scored.resize(topn); return scored;
}

vector<int> lexical_recall(const vector<string>& qtokens, int topn=100){ vector<pair<int,float>> scored; for(size_t i=0;i<segments.size();++i){ float sc = bm25_expected_score(qtokens, segments[i]); if(sc>0) scored.push_back({(int)i, sc}); } sort(scored.begin(), scored.end(), [](auto&a,auto&b){return a.second>b.second;}); vector<int> out; for(size_t i=0;i<scored.size() && (int)out.size()<topn;++i) out.push_back(scored[i].first); return out; }

// cross-encoder placeholder
float cross_encoder_score_placeholder(const string& query, const string& segment_text){ if(segment_text.empty()) return 0.0f; size_t common=0; for(size_t i=0;i<query.size();++i){ for(size_t j=0;j<segment_text.size();++j){ size_t k=0; while(i+k<query.size() && j+k<segment_text.size() && query[i+k]==segment_text[j+k]) ++k; common = max(common, k); } } return (float)common / (float)max((size_t)1, segment_text.size()); }

struct Candidate{ int video_id; float s_ce; float personal; float bm25; float sem; float ph; float fresh; float base_score; vector<float> embedding; };

vector<int> greedy_diverse_select(vector<Candidate>& candidates, int k=10, float diversity_penalty=0.8f){ vector<int> selected_ids; vector<vector<float>> sel_embs; sort(candidates.begin(), candidates.end(), [](auto&a,auto&b){return a.base_score>b.base_score;}); while((int)selected_ids.size()<k && !candidates.empty()){ int best_idx=-1; float best_score=-1e9; for(size_t i=0;i<candidates.size();++i){ float penalty=0; for(auto &se: sel_embs){ float sim = cosine_sim(se, candidates[i].embedding); if(sim>penalty) penalty=sim; } float penalized = candidates[i].base_score - diversity_penalty*penalty; if(penalized>best_score){ best_score=penalized; best_idx=(int)i; } } if(best_idx==-1) break; selected_ids.push_back(candidates[best_idx].video_id); sel_embs.push_back(candidates[best_idx].embedding); candidates.erase(candidates.begin()+best_idx); } return selected_ids; }

void build_demo_dataset(){
    videos.clear(); segments.clear();
    for(int i=0;i<5;++i){ Video v; v.id=i; v.title="Video_"+to_string(i); v.created_ts = 1e9f - i*100000; v.duration=30.0f; v.embedding = vector<float>(64); for(size_t j=0;j<v.embedding.size();++j) v.embedding[j]=float(((i+1)*(j+1))%10)/10.0f; v.embedding = normalize(v.embedding); videos.push_back(v); }
    // segments
    Segment s0; s0.id=0; s0.video_id=0; s0.start=0; s0.end=4; s0.transcript="hello world"; s0.asr_confidence=0.9; s0.asr_lattice={{"hello world", -1.0f},{"yellow world", -3.0f}}; segments.push_back(s0); videos[0].segment_ids.push_back(0);
    Segment s1; s1.id=1; s1.video_id=1; s1.start=1; s1.end=5; s1.transcript="buy this product now"; s1.asr_confidence=0.85; s1.asr_lattice={{"buy this product now", -0.5f},{"buy cheap product", -2.0f}}; segments.push_back(s1); videos[1].segment_ids.push_back(1);
    Segment s2; s2.id=2; s2.video_id=2; s2.start=0.5; s2.end=3.5; s2.transcript="special discount today"; s2.asr_confidence=0.8; s2.asr_lattice={{"special discount today", -0.8f}}; segments.push_back(s2); videos[2].segment_ids.push_back(2);
    Segment s3; s3.id=3; s3.video_id=3; s3.start=2; s3.end=6; s3.transcript="new summer collection"; s3.asr_confidence=0.7; s3.asr_lattice={{"new summer collection", -0.6f}}; segments.push_back(s3); videos[3].segment_ids.push_back(3);
}

int main(){
    cout<<"Recommender demo starting..."<<endl;
    build_demo_dataset();
    // build FAISS index if available
    FaissIndexWrapper faissWrap(64);
    vector<vector<float>> allEmb; for(auto &v: videos) allEmb.push_back(v.embedding);
    faissWrap.add(allEmb);

    string query = "buy product";
    vector<string> qtokens; { stringstream ss(query); string w; while(ss>>w) qtokens.push_back(w); }

    // dense recall via FAISS (if available) else brute-force
    vector<pair<int,float>> denseRes;
#ifdef HAVE_FAISS
    denseRes = faissWrap.search(vector<float>(64,0.1f), 10); // demo query vector placeholder
    if(denseRes.empty()) denseRes = dense_recall_bruteforce(vector<float>(64,0.1f), 10);
#else
    denseRes = dense_recall_bruteforce(vector<float>(64,0.1f), 10);
#endif
    cout<<"Dense recall returned "<<denseRes.size()<<" items"<<endl;

    auto lexSegs = lexical_recall(qtokens, 50);
    cout<<"Lexical recall segments: "; for(auto s: lexSegs) cout<<s<<" "; cout<<endl;

    // candidate videos union
    unordered_set<int> cand_vids;
    for(auto &p: denseRes) cand_vids.insert(p.first);
    for(auto sid: lexSegs) cand_vids.insert(segments[sid].video_id);

    vector<Candidate> candidates;
    for(int vid: cand_vids){ auto &v = videos[vid]; float bestTokenProb=0; int bestSeg=-1; for(auto sid: v.segment_ids){ float sumProb=0; for(auto &tok: qtokens) sumProb += token_present_prob(tok, segments[sid]); if(sumProb>bestTokenProb){ bestTokenProb=sumProb; bestSeg=sid; } }
        string bestText = bestSeg>=0 ? segments[bestSeg].transcript : "";
        float bm25 = bestSeg>=0 ? bm25_expected_score(qtokens, segments[bestSeg]) : 0.0f;
        float sem = cosine_sim(vector<float>(64,0.1f), v.embedding); // placeholder
        float s_ce = cross_encoder_score_placeholder(query, bestText);
        float personal = cosine_sim(vector<float>(64,0.2f), v.embedding); // placeholder user vector
        Candidate c; c.video_id = v.id; c.s_ce = s_ce; c.personal = personal; c.bm25 = bm25; c.sem = sem; c.ph=0; c.fresh=1.0f; c.embedding=v.embedding; c.base_score = 1.0f*s_ce + 1.0f*personal + 1.0f*bm25 + 1.0f*sem;
        candidates.push_back(c);
    }

    auto selected = greedy_diverse_select(candidates, 3, 0.7f);
    cout<<"Selected videos: "; for(auto id: selected) cout<<id<<" "; cout<<"\n";

    cout<<"Detailed candidates:\n";
    for(auto &c: candidates) cout<<"vid="<<c.video_id<<" base="<<c.base_score<<" s_ce="<<c.s_ce<<" personal="<<c.personal<<" bm25="<<c.bm25<<" sem="<<c.sem<<"\n";

    return 0;
}
