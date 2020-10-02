//leetcode第46题，使用了了交换法解决全排列问题，比回溯更快
class Solution {
public:
    vector<vector<int>> ans;//加个start真是神技
    void backtrack(vector<int>& nums,int first)
    {
        if(first==nums.size())
        {

            ans.push_back(nums);
            return;

        }
            for(int i=first;i<nums.size();i++)
            {
                swap(nums[first],nums[i]);
                backtrack(nums,first+1);
                swap(nums[first],nums[i]);
            }
        }
    
    vector<vector<int>> permute(vector<int>& nums) {
        if(nums.empty())return ans;
        backtrack(nums,0);
        return ans;
        
    }
};
