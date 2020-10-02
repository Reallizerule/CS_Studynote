class Solution {
public:
    vector<bool> kidsWithCandies(vector<int>& candies, int extraCandies) {
        int max = 0;
        vector <bool> res ( candies.size() , false);
        for ( int i = 0 ; i < candies.size() ; i++ ){
            if ( candies[i] > max ){
                max = candies[i];
            }
        }
        for ( int i = 0 ; i < candies.size() ; i++ ){
            if ( candies[i] + extraCandies >= max ){
                res[i] = true;
            }
        }
        return res;
    }
};
