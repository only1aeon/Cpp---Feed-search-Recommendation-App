#pragma once
#include <vector>
#include <string>
using namespace std;
float dot(const vector<float>& a, const vector<float>& b);
float norm(const vector<float>& a);
vector<float> normalize(const vector<float>& a);
float cosine_sim(const vector<float>& a, const vector<float>& b);
