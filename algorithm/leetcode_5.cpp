//软院c语言课程课堂作业
//最长回文子串
#include<iostream>
#include<string>
#include<algorithm>
using namespace std;
string help(string& s, int l, int r) {
    while (l >= 0 && r < s.size()
            && s[l] == s[r]) {
        l--; r++;
    }
    return s.substr(l + 1, r - l - 1);
}

string longest(string s) {
    string res;
    for (int i = 0; i < s.size(); i++) {
        string s1 = help(s, i, i);
        string s2 = help(s, i, i + 1);
        res = res.size() > s1.size() ? res : s1;
        res = res.size() > s2.size() ? res : s2;
    }
    return res;
}


int main()
{
	string s;
    int tmp;
    cin>>tmp;
	cin>>s;
	string s2=longest(s);
	int len=s2.size();
	cout<<len;
	return 0;

	
}
