//本题为leetcode 200.岛屿数目
//其题目与软院开学考试第二题基本一致
class Solution {
public:
    int ans=0;
    void dfs(vector<vector<char>>& grid,int r,int c)
    {
        int row=grid.size();
        int col=grid[0].size();
        grid[r][c]='0';
        if(r-1>=0&&grid[r-1][c]=='1')dfs(grid,r-1,c);
        if(r+1<row&&grid[r+1][c]=='1')dfs(grid,r+1,c);
        if(c-1>=0&&grid[r][c-1]=='1')dfs(grid,r,c-1);
        if(c+1<col&&grid[r][c+1]=='1')dfs(grid,r,c+1);
    }
    int numIslands(vector<vector<char>>& grid) {

        int row=grid.size();
        if(row==0)return ans;
        int col=grid[0].size();
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                if(grid[i][j]=='1')
                {
                    ans++;
                    dfs(grid,i,j);
                }
            }
        }
        return ans;

    }
};
