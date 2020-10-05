//题目为三数之和，是两数之和那道题的延申
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> temp;
        
        sort(nums.begin(), nums.end());
        
        map<int,int> map;
        for (int i=0; i<nums.size(); i++) {
            map[nums[i]] = i;
        }
        int index = INT32_MIN;
        for (int i=0; i<nums.size(); i++) {
            if (nums[i]>0) {
                break;
            }
            if (index == nums[i]) {
                continue;
            } else{
                index = nums[i];
            }
            int sum = -nums[i];
            
            int index1 = INT32_MIN;
            for (int j=i+1; j<nums.size(); j++) {
                if (index1 == nums[j]) {
                    continue;
                } else{
                    index1 = nums[j];
                }
                int search = sum - nums[j];
                if (map.find(search) != map.end() && map[search] > i && map[search] > j) {
                    vector<int> vec = {nums[i],nums[j],search};
                    temp.push_back(vec);
                }
            }
            
        }
        return temp;
    }
};
