#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
int helper(vector<int> ve)
{
	int len=ve.size();
//使用动态规划的方法，用空间换时间
	vector<int> dp(len,0);
	dp[0]=ve[0];
	for(int i=1;i<len;i++)
	{
		dp[i]=max(ve[i],dp[i-1]+ve[i]);
	}
	int maxans=0;
	for(int i=0;i<len;i++)
	{
		maxans=max(dp[i],maxans);
	}
	return maxans;

}

int main()
{
	int n;
	while(cin>>n)
	{
		if(n==0)break;
		int ans=0;
		vector<int> vec;
		for(int i=0;i<n;i++)
		{
			int tmp;
			cin>>tmp;
			vec.push_back(tmp);
		}
		int sum;
		sum=helper(vec);
		cout<<sum<<endl;
	}
	return 0;
  
}
