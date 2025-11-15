#include "utils.h"
#include <cmath>
float dot(const vector<float>& a, const vector<float>& b){ float s=0; size_t n=min(a.size(), b.size()); for(size_t i=0;i<n;++i) s+=a[i]*b[i]; return s; }
float norm(const vector<float>& a){ return sqrtf(max(0.0f, dot(a,a))); }
vector<float> normalize(const vector<float>& a){ float n=norm(a)+1e-9f; vector<float> out(a.size()); for(size_t i=0;i<a.size();++i) out[i]=a[i]/n; return out; }
float cosine_sim(const vector<float>& a, const vector<float>& b){ return dot(a,b)/( (norm(a)+1e-9f)*(norm(b)+1e-9f) ); }
